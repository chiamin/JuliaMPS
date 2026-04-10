using Test
using LinearAlgebra
using TensorOperations
using KrylovKit

include("../src/Tensor/TensorUtils.jl")
include("../src/Sites/Site.jl")
include("../src/Sites/SpinHalf.jl")
include("../src/MPS/MPS.jl")
include("../src/MPS/MPO.jl")
include("../src/MPS/Init.jl")
include("../src/MPS/Operations.jl")
include("../src/MPS/AutoMPO.jl")
include("../src/DMRG/Environment.jl")
include("../src/DMRG/EffectiveOperators.jl")
include("../src/DMRG/Engine.jl")
include("../src/TDVP/Engine.jl")

function mpo_to_matrix(mpo::MPO{T}) where {T}
    N = length(mpo)
    d = size(mpo[1], 2)
    block = copy(mpo[1])
    for k in 2:N
        Wk = mpo[k]
        sz_b = size(block); sz_w = size(Wk)
        bmat = reshape(block, prod(sz_b[1:end-1]), sz_b[end])
        wmat = reshape(Wk,    sz_w[1], sz_w[2]*sz_w[3]*sz_w[4])
        cmat = bmat * wmat
        block = reshape(cmat, sz_b[1:end-1]..., sz_w[2], sz_w[3], sz_w[4])
    end
    ip_dims = [2*k     for k in 1:N]
    i_dims  = [2*k + 1 for k in 1:N]
    perm    = [reverse(ip_dims); reverse(i_dims); 1; 2*N + 2]
    block_p = permutedims(block, perm)
    return reshape(block_p, d^N, d^N)
end

function heisenberg_mpo(N::Int, J::Float64=1.0)
    site = spin_half()
    ampo = AutoMPO(N, site)
    for i in 1:N-1
        add!(ampo, J/2, "Sp", i, "Sm", i+1)
        add!(ampo, J/2, "Sm", i, "Sp", i+1)
        add!(ampo, J,   "Sz", i, "Sz", i+1)
    end
    return to_mpo(ampo)
end

# ---------------------------------------------------------------------------
# Imaginary-time TDVP → ground state energy
# ---------------------------------------------------------------------------

@testset "Imaginary-time 2-site TDVP → ground state" begin
    N = 4
    H = heisenberg_mpo(N)
    e_ed = minimum(eigvals(Hermitian(mpo_to_matrix(H))))

    psi = random_mps(Float64, N, 2, 8)
    orthogonalize!(psi)
    engine = TDVPEngine(psi, H)
    # Use a small step + many iterations to keep Trotter error small.
    dt = 0.05
    for _ in 1:200
        sweep!(engine, dt; num_center=2, max_dim=16, cutoff=1e-12)
        LinearAlgebra.normalize!(psi)
    end
    E = real(expectation(psi, H, psi)) / real(inner(psi, psi))
    # TDVP imaginary time converges with O(dt^2) per-step error;
    # accept ~1e-4 absolute for this step size and N.
    @test isapprox(E, e_ed; atol=1e-4)
end

@testset "Imaginary-time 1-site TDVP → ground state" begin
    N = 4
    H = heisenberg_mpo(N)
    e_ed = minimum(eigvals(Hermitian(mpo_to_matrix(H))))

    psi = random_mps(Float64, N, 2, 16)   # need adequate D from start
    orthogonalize!(psi)
    engine = TDVPEngine(psi, H)
    dt = 0.05
    for _ in 1:200
        sweep!(engine, dt; num_center=1)
        LinearAlgebra.normalize!(psi)
    end
    E = real(expectation(psi, H, psi)) / real(inner(psi, psi))
    @test isapprox(E, e_ed; atol=1e-3)
end

# ---------------------------------------------------------------------------
# Real-time TDVP: energy and norm conservation
# ---------------------------------------------------------------------------

@testset "Real-time 2-site TDVP conserves energy and norm" begin
    N = 4
    H = heisenberg_mpo(N)
    H_complex = MPO([ComplexF64.(H[k]) for k in 1:N])

    psi = random_mps(ComplexF64, N, 2, 8)
    orthogonalize!(psi)
    LinearAlgebra.normalize!(psi)

    E0 = real(expectation(psi, H_complex, psi))
    n0 = norm(psi)

    engine = TDVPEngine(psi, H_complex)
    dt = 1im * 0.05   # real-time step
    for _ in 1:10
        sweep!(engine, dt; num_center=2, max_dim=16, cutoff=1e-12)
    end

    E1 = real(expectation(psi, H_complex, psi))
    n1 = norm(psi)
    @test isapprox(E1, E0; atol=1e-6)
    @test isapprox(n1, n0; atol=1e-8)
end

@testset "Real-time 1-site TDVP conserves energy and norm" begin
    N = 4
    H = heisenberg_mpo(N)
    H_complex = MPO([ComplexF64.(H[k]) for k in 1:N])

    psi = random_mps(ComplexF64, N, 2, 16)
    orthogonalize!(psi)
    LinearAlgebra.normalize!(psi)

    E0 = real(expectation(psi, H_complex, psi))
    n0 = norm(psi)

    engine = TDVPEngine(psi, H_complex)
    dt = 1im * 0.05
    for _ in 1:10
        sweep!(engine, dt; num_center=1)
    end

    E1 = real(expectation(psi, H_complex, psi))
    n1 = norm(psi)
    @test isapprox(E1, E0; atol=1e-7)
    @test isapprox(n1, n0; atol=1e-8)
end
