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

# ED helper: build dense H from MPO (kron-ordered)
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

@testset "DMRG Heisenberg N=4 ground state (2-site)" begin
    N = 4
    H = heisenberg_mpo(N)
    H_mat = mpo_to_matrix(H)
    e_ed = minimum(eigvals(Hermitian(H_mat)))

    psi = random_mps(Float64, N, 2, 4)
    orthogonalize!(psi)   # center=1
    energies = dmrg!(psi, H; sweeps=4, max_dims=[10,20,20,20], cutoffs=[0,0,1e-10,1e-10])

    @test isapprox(energies[end], e_ed; atol=1e-8)
end

@testset "DMRG Heisenberg N=8 ground state (2-site)" begin
    N = 8
    H = heisenberg_mpo(N)
    H_mat = mpo_to_matrix(H)
    e_ed = minimum(eigvals(Hermitian(H_mat)))

    psi = random_mps(Float64, N, 2, 8)
    orthogonalize!(psi)
    energies = dmrg!(psi, H; sweeps=6, max_dims=[10,20,30,30,30,30],
                     cutoffs=[0,0,0,1e-10,1e-10,1e-10])

    @test isapprox(energies[end], e_ed; atol=1e-8)
end

@testset "DMRG Heisenberg N=4 ground state (1-site)" begin
    N = 4
    H = heisenberg_mpo(N)
    H_mat = mpo_to_matrix(H)
    e_ed = minimum(eigvals(Hermitian(H_mat)))

    psi = random_mps(Float64, N, 2, 8)   # need adequate bond dim from start (1-site can't grow)
    orthogonalize!(psi)
    engine = DMRGEngine(psi, H)
    local E
    for _ in 1:8
        E, _ = sweep!(engine; num_center=1)
    end
    @test isapprox(E, e_ed; atol=1e-6)
end

@testset "DMRG complex Hamiltonian energy is real" begin
    N = 4
    H_real = heisenberg_mpo(N)
    # Promote MPO to ComplexF64
    H_complex_tensors = [ComplexF64.(H_real[k]) for k in 1:N]
    H = MPO(H_complex_tensors)
    H_mat = mpo_to_matrix(H)
    e_ed = minimum(real.(eigvals(Hermitian(H_mat))))

    psi = random_mps(ComplexF64, N, 2, 4)
    orthogonalize!(psi)
    energies = dmrg!(psi, H; sweeps=4, max_dims=[10,20,20,20], cutoffs=[0,0,1e-10,1e-10])
    @test isapprox(energies[end], e_ed; atol=1e-8)
end

@testset "DMRG excited state via penalty" begin
    N = 6
    H = heisenberg_mpo(N)
    H_mat = mpo_to_matrix(H)
    eigs_ed = sort(eigvals(Hermitian(H_mat)))
    e0_ed, e1_ed = eigs_ed[1], eigs_ed[2]

    # Ground state
    psi0 = random_mps(Float64, N, 2, 8)
    orthogonalize!(psi0)
    energies0 = dmrg!(psi0, H; sweeps=5, max_dims=fill(20, 5), cutoffs=fill(1e-10, 5))
    @test isapprox(energies0[end], e0_ed; atol=1e-7)

    # First excited via orthogonality penalty
    psi1 = random_mps(Float64, N, 2, 8)
    orthogonalize!(psi1)
    engine = DMRGEngine(psi1, H; ortho_states=[psi0], ortho_weights=[100.0])
    local E1
    for _ in 1:8
        E1, _ = sweep!(engine; max_dim=20, cutoff=1e-10, num_center=2)
    end
    @test isapprox(E1, e1_ed; atol=1e-6)
end
