using Test
using LinearAlgebra
using TensorOperations

include("../src/Tensor/TensorUtils.jl")
include("../src/Sites/Site.jl")
include("../src/Sites/SpinHalf.jl")
include("../src/Sites/SpinlessFermion.jl")
include("../src/MPS/MPS.jl")
include("../src/MPS/MPO.jl")
include("../src/MPS/Init.jl")
include("../src/MPS/Operations.jl")
include("../src/MPS/AutoMPO.jl")

# ---------------------------------------------------------------------------
# ED helpers
# ---------------------------------------------------------------------------

# Build the full 2^N × 2^N matrix from a single-site operator at site k.
function single_site_op(::Type{T}, mat::AbstractMatrix, k::Int, N::Int, d::Int) where {T}
    Id = Matrix{T}(I, d, d)
    M  = T.(mat)
    out = k == 1 ? M : Id
    for j in 2:N
        out = kron(out, j == k ? M : Id)
    end
    return out
end

# Two-site product op (with possible JW string handled by caller).
function two_site_op(::Type{T}, A::AbstractMatrix, ka::Int,
                                B::AbstractMatrix, kb::Int, N::Int, d::Int) where {T}
    Id = Matrix{T}(I, d, d)
    pieces = [Id for _ in 1:N]
    pieces[ka] = T.(A)
    pieces[kb] = T.(B)
    out = pieces[1]
    for j in 2:N
        out = kron(out, pieces[j])
    end
    return out
end

# Build dense Hamiltonian matrix by contracting all MPO tensors.
#
# The convention to match `kron(A, B, ...)`:
#   kron's linearized row index = ip1 * d^(N-1) + ip2 * d^(N-2) + ... + ipN
#   so ip1 is the SLOWEST-changing index.
# Julia's column-major `reshape` makes the FIRST dim the fastest, so we
# permute to (ipN, ip_{N-1}, ..., ip1, iN, ..., i1) before reshaping.
function mpo_to_matrix(mpo::MPO{T}) where {T}
    N = length(mpo)
    d = size(mpo[1], 2)
    # Iteratively contract: keep `block` with leading dim = old left bond (=1),
    # appending (ip_k, i_k) per site, trailing dim = current right bond.
    block = copy(mpo[1])   # shape (1, ip1, i1, r1)
    for k in 2:N
        Wk = mpo[k]
        sz_b = size(block)
        sz_w = size(Wk)
        bmat = reshape(block, prod(sz_b[1:end-1]), sz_b[end])
        wmat = reshape(Wk,    sz_w[1], sz_w[2]*sz_w[3]*sz_w[4])
        cmat = bmat * wmat
        block = reshape(cmat, sz_b[1:end-1]..., sz_w[2], sz_w[3], sz_w[4])
    end
    # block shape: (1, ip1, i1, ip2, i2, ..., ipN, iN, 1)
    ip_dims = [2*k     for k in 1:N]   # 2, 4, ..., 2N
    i_dims  = [2*k + 1 for k in 1:N]   # 3, 5, ..., 2N+1
    perm    = [reverse(ip_dims); reverse(i_dims); 1; 2*N + 2]
    block_p = permutedims(block, perm)
    return reshape(block_p, d^N, d^N)
end

# ---------------------------------------------------------------------------
# Heisenberg N=4
# ---------------------------------------------------------------------------

@testset "AutoMPO Heisenberg N=4" begin
    site = spin_half()
    N = 4
    J = 1.0
    ampo = AutoMPO(N, site)
    for i in 1:N-1
        add!(ampo, J/2, "Sp", i, "Sm", i+1)
        add!(ampo, J/2, "Sm", i, "Sp", i+1)
        add!(ampo, J,   "Sz", i, "Sz", i+1)
    end
    H = to_mpo(ampo)
    @test length(H) == N

    # Build the same H by ED and compare
    Sz = real(op(site, "Sz")); Sp = real(op(site, "Sp")); Sm = real(op(site, "Sm"))
    H_ed = zeros(Float64, 2^N, 2^N)
    for i in 1:N-1
        H_ed .+= (J/2) .* two_site_op(Float64, Sp, i, Sm, i+1, N, 2)
        H_ed .+= (J/2) .* two_site_op(Float64, Sm, i, Sp, i+1, N, 2)
        H_ed .+= J     .* two_site_op(Float64, Sz, i, Sz, i+1, N, 2)
    end

    H_mpo_mat = mpo_to_matrix(H)
    @test isapprox(H_mpo_mat, H_ed; atol=1e-12)

    # Spectrum should match
    eig_mpo = sort(real.(eigen(Hermitian(H_mpo_mat)).values))
    eig_ed  = sort(eigvals(Hermitian(H_ed)))
    @test isapprox(eig_mpo, eig_ed; atol=1e-12)
end

# ---------------------------------------------------------------------------
# Spinless fermion tight-binding N=4 (with JW)
# ---------------------------------------------------------------------------

@testset "AutoMPO spinless fermion tight-binding N=4" begin
    site = spinless_fermion()
    N = 4
    t = 1.0
    ampo = AutoMPO(N, site)
    for i in 1:N-1
        add!(ampo, -t, "Cdag", i,   "C", i+1)
        add!(ampo, -t, "Cdag", i+1, "C", i)
    end
    H = to_mpo(ampo)
    @test length(H) == N

    # ED: build with JW manually
    # Physical fermion operators in Fock basis (basis 1=|0>, 2=|1>):
    # c_k        = (F ⊗ F ⊗ ... ⊗ F ⊗ a) ⊗ (I ⊗ I ⊗ ...)   (F at sites 1..k-1, a at k)
    # cdag_k     = (F ⊗ F ⊗ ... ⊗ F ⊗ a†) ⊗ (I ⊗ I ⊗ ...)
    a    = real(op(site, "C"))
    adag = real(op(site, "Cdag"))
    F    = real(op(site, "F"))
    Id   = Matrix{Float64}(I, 2, 2)

    function fermion_op(k::Int, local_op::Matrix{Float64})
        out = (k == 1) ? local_op : F
        for j in 2:N
            piece = j < k ? F : (j == k ? local_op : Id)
            out = kron(out, piece)
        end
        return out
    end

    H_ed = zeros(Float64, 2^N, 2^N)
    for i in 1:N-1
        ci    = fermion_op(i,   a)
        cidag = fermion_op(i,   adag)
        cj    = fermion_op(i+1, a)
        cjdag = fermion_op(i+1, adag)
        H_ed .+= -t .* (cidag * cj)
        H_ed .+= -t .* (cjdag * ci)
    end

    H_mpo_mat = mpo_to_matrix(H)
    @test isapprox(H_mpo_mat, H_ed; atol=1e-12)
end

# ---------------------------------------------------------------------------
# Long-range fermion: c+_1 c_4 should pick up F at sites 2,3
# ---------------------------------------------------------------------------

@testset "AutoMPO long-range fermion JW string" begin
    site = spinless_fermion()
    N = 4
    ampo = AutoMPO(N, site)
    add!(ampo, 1.0, "Cdag", 1, "C", 4)
    H = to_mpo(ampo)

    a    = real(op(site, "C"))
    adag = real(op(site, "Cdag"))
    F    = real(op(site, "F"))
    Id   = Matrix{Float64}(I, 2, 2)

    function fermion_op(k::Int, local_op::Matrix{Float64})
        out = (k == 1) ? local_op : F
        for j in 2:N
            piece = j < k ? F : (j == k ? local_op : Id)
            out = kron(out, piece)
        end
        return out
    end

    H_ed = fermion_op(1, adag) * fermion_op(4, a)
    H_mpo_mat = mpo_to_matrix(H)
    @test isapprox(H_mpo_mat, H_ed; atol=1e-12)
end

# ---------------------------------------------------------------------------
# expectation(psi, H, psi) sanity vs ED for a random product state
# ---------------------------------------------------------------------------

@testset "AutoMPO expectation matches ED" begin
    site = spin_half()
    N = 4
    J = 0.7
    ampo = AutoMPO(N, site)
    for i in 1:N-1
        add!(ampo, J,   "Sz", i, "Sz", i+1)
        add!(ampo, J/2, "Sp", i, "Sm", i+1)
        add!(ampo, J/2, "Sm", i, "Sp", i+1)
    end
    H = to_mpo(ampo)

    psi = random_mps(Float64, N, 2, 4)
    # Reshape MPS to a full state vector by exact contraction.
    # Same column-major / kron-ordering caveat as mpo_to_matrix.
    function mps_to_vec(psi::MPS{T}) where {T}
        N = length(psi)
        d = size(psi[1], 2)
        block = copy(psi[1])   # (1, i1, r1)
        for k in 2:N
            A = psi[k]
            sz_b = size(block)
            sz_a = size(A)
            bmat = reshape(block, prod(sz_b[1:end-1]), sz_b[end])
            amat = reshape(A,     sz_a[1], sz_a[2]*sz_a[3])
            cmat = bmat * amat
            block = reshape(cmat, sz_b[1:end-1]..., sz_a[2], sz_a[3])
        end
        # block shape: (1, i1, i2, ..., iN, 1)
        i_dims = [k + 1 for k in 1:N]   # 2, 3, ..., N+1
        perm = [reverse(i_dims); 1; N + 2]
        block_p = permutedims(block, perm)
        return reshape(block_p, d^N)
    end

    v = mps_to_vec(psi)
    H_ed = mpo_to_matrix(H)
    e_ed  = real(v' * H_ed * v)
    e_mpo = real(expectation(psi, H, psi))
    @test isapprox(e_mpo, e_ed; atol=1e-10)
end
