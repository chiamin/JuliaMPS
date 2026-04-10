using Test
using LinearAlgebra
using TensorOperations

include("../src/Tensor/TensorUtils.jl")
include("../src/MPS/MPS.jl")
include("../src/MPS/Init.jl")

@testset "random_mps" begin
    for T in (Float64, ComplexF64)
        psi = random_mps(T, 6, 2, 4)
        @test length(psi) == 6
        @test eltype(psi) == T
        # Boundary bonds dim 1; interior bonds <= 4 (may be smaller after
        # orthogonalize+normalize when phys^k caps the bond).
        bd = bond_dims(psi)
        @test bd[1] == 1 && bd[end] == 1
        @test maximum(bd) <= 4
        @test isapprox(norm(psi), 1.0; atol=1e-10)
        @test center(psi) !== nothing
    end

    # Without normalize
    psi = random_mps(Float64, 4, 2, 3; normalize=false)
    @test psi.center_left == 1
    @test psi.center_right == 4

    # 1-site degenerate case
    psi1 = random_mps(Float64, 1, 2, 4)
    @test length(psi1) == 1
    @test bond_dims(psi1) == [1, 1]

    # Bad inputs
    @test_throws ErrorException random_mps(Float64, 0, 2, 3)
    @test_throws ErrorException random_mps(Float64, 3, 0, 3)
    @test_throws ErrorException random_mps(Float64, 3, 2, 0)
end

@testset "product_state" begin
    # Spin-up product state on 4 sites
    up = [1.0, 0.0]
    states = [up, up, up, up]
    psi = product_state(Float64, states)
    @test length(psi) == 4
    @test bond_dims(psi) == [1,1,1,1,1]

    # <psi|psi> = 1
    nrm_sq = sum(abs2.(psi[1])) * sum(abs2.(psi[2])) *
             sum(abs2.(psi[3])) * sum(abs2.(psi[4]))
    @test isapprox(nrm_sq, 1.0; atol=1e-12)

    # ComplexF64
    psi_c = product_state(ComplexF64, [up, up])
    @test eltype(psi_c) == ComplexF64
end
