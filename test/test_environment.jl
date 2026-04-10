using Test
using LinearAlgebra
using TensorOperations

include("../src/Tensor/TensorUtils.jl")
include("../src/MPS/MPS.jl")
include("../src/MPS/MPO.jl")
include("../src/MPS/Init.jl")
include("../src/MPS/Operations.jl")
include("../src/DMRG/Environment.jl")

function make_random_mpo(::Type{T}, N::Int, d::Int, D::Int) where {T<:Number}
    md = [1; fill(D, N-1); 1]
    tensors = [randn(T, md[k], d, d, md[k+1]) for k in 1:N]
    return MPO(tensors)
end

@testset "OperatorEnv boundary and full contraction" begin
    for T in (Float64, ComplexF64)
        N = 5
        psi = random_mps(T, N, 2, 3)
        W   = make_random_mpo(T, N, 2, 3)

        env = OperatorEnv(psi, psi, W; init_center=1)
        @test env.centerL == 1 && env.centerR == 1

        # Boundaries should still be ones
        @test env.LR[0]   == ones(T, 1, 1, 1)
        @test env.LR[N+1] == ones(T, 1, 1, 1)

        # After update_envs!(env, N, N), LR[0..N-1] should be valid left envs;
        # contracting LR[N-1] with site N and LR[N+1] should give <psi|W|psi>.
        update_envs!(env, N, N)
        @test env.centerL == N && env.centerR == N

        Lend = getenv(env, N - 1)   # left env covering sites 1..N-1
        Rend = getenv(env, N + 1)   # right boundary
        ket  = psi[N]
        Wend = W[N]
        bra  = psi[N]
        @tensor scalar[] := Lend[xdn, xw, xup] *
                            ket[xdn, i, ydn] *
                            Wend[xw, ip, i, yw] *
                            conj(bra)[xup, ip, yup] *
                            Rend[ydn, yw, yup]
        manual = scalar[]
        @test isapprox(manual, expectation(psi, W, psi); atol=1e-10)
    end
end

@testset "OperatorEnv update_envs grow both sides" begin
    T = Float64
    N = 6
    psi = random_mps(T, N, 2, 3)
    W   = make_random_mpo(T, N, 2, 3)

    env = OperatorEnv(psi, psi, W; init_center=3)
    @test env.centerL == 3 && env.centerR == 3

    # Move stale window to [1,1]: grows right envs
    update_envs!(env, 1, 1)
    @test env.centerL == 1 && env.centerR == 1

    # Move stale window back to [N,N]: grows left envs
    update_envs!(env, N, N)
    @test env.centerL == N && env.centerR == N
end

@testset "OperatorEnv invalidation via callback" begin
    T = Float64
    N = 5
    psi = random_mps(T, N, 2, 3)
    W   = make_random_mpo(T, N, 2, 3)

    env = OperatorEnv(psi, psi, W; init_center=3)
    # After init: centerL=centerR=3; LR[2] is valid
    update_envs!(env, 2, 2)
    @test env.centerL == 2

    # Modify site 1 → invalidates LR[1]
    psi[1] = copy(psi[1])
    @test env.centerL == 1
end

@testset "VectorEnv overlap" begin
    for T in (Float64, ComplexF64)
        N = 5
        psi = random_mps(T, N, 2, 3)
        phi = random_mps(T, N, 2, 3)

        env = VectorEnv(phi, psi; init_center=1)
        update_envs!(env, N, N)
        Lend = getenv(env, N - 1)
        Rend = getenv(env, N + 1)
        ket  = phi[N]
        bra  = psi[N]
        @tensor scalar[] :=
            Lend[xdn, xup] *
            ket[xdn, i, ydn] *
            conj(bra)[xup, i, yup] *
            Rend[ydn, yup]
        @test isapprox(scalar[], inner(psi, phi); atol=1e-10)
    end
end
