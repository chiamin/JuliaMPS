using Test
using LinearAlgebra
using TensorOperations

include("../src/Tensor/TensorUtils.jl")
include("../src/MPS/MPS.jl")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

"""Build a random MPS with the given element type, num_sites, phys_dim, bond_dim."""
function make_random_mps(::Type{T}, N::Int, d::Int, D::Int) where {T<:Number}
    bd = [1; fill(D, N-1); 1]
    tensors = [randn(T, bd[k], d, bd[k+1]) for k in 1:N]
    return MPS(tensors)
end

# Direct contraction of two MPS into a scalar <bra|ket>
function direct_inner(bra::MPS{T}, ket::MPS{T}) where {T}
    env = ones(T, 1, 1)
    for k in 1:length(ket)
        K = ket[k]
        B = bra[k]
        @tensor env_new[dn, up] := env[x, y] * K[x, i, dn] * conj(B)[y, i, up]
        env = env_new
    end
    return only(env)
end

# ---------------------------------------------------------------------------
# Construction & basic properties
# ---------------------------------------------------------------------------

@testset "MPS construction" begin
    psi = make_random_mps(Float64, 6, 2, 4)

    @test length(psi) == 6
    @test eltype(psi) == Float64
    @test phys_dims(psi) == [2,2,2,2,2,2]
    @test bond_dims(psi) == [1,4,4,4,4,4,1]
    @test max_dim(psi) == 4

    # Default center window covers everything
    @test psi.center_left == 1
    @test psi.center_right == 6
    @test center(psi) === nothing

    # Boundary check
    @test size(psi[1], 1) == 1
    @test size(psi[end], 3) == 1

    # Bad rank: Vector{Matrix} doesn't even match the constructor signature
    @test_throws MethodError MPS([randn(2, 2)])

    # Bond mismatch should error (3 vs 2 between sites)
    bad = [randn(1, 2, 3), randn(2, 2, 2), randn(2, 2, 1)]
    @test_throws ErrorException MPS(bad)

    # Boundary left != 1
    bad2 = [randn(2, 2, 1)]
    @test_throws ErrorException MPS(bad2)
end

# ---------------------------------------------------------------------------
# norm / normalize!
# ---------------------------------------------------------------------------

@testset "norm and normalize!" begin
    for T in (Float64, ComplexF64)
        psi = make_random_mps(T, 5, 2, 3)
        n0 = norm(psi)
        @test n0 > 0
        @test isapprox(n0^2, real(direct_inner(psi, psi)); atol=1e-10)

        # Cannot normalize without single center
        @test_throws ErrorException normalize!(psi)

        orthogonalize!(psi)
        @test center(psi) !== nothing
        normalize!(psi)
        @test isapprox(norm(psi), 1.0; atol=1e-10)
    end
end

# ---------------------------------------------------------------------------
# orthogonalize! and move_center!
# ---------------------------------------------------------------------------

@testset "orthogonalize! and move_center!" begin
    for T in (Float64, ComplexF64)
        psi = make_random_mps(T, 6, 2, 4)
        n_before = norm(psi)

        orthogonalize!(psi)
        @test psi.center_left == 1 && psi.center_right == 1
        @test isapprox(norm(psi), n_before; atol=1e-10)
        check_left_right_orthonormal(psi; atol=1e-10)

        move_center!(psi, 4)
        @test center(psi) == 4
        check_left_right_orthonormal(psi; atol=1e-10)
        @test isapprox(norm(psi), n_before; atol=1e-10)

        move_center!(psi, 6)
        @test center(psi) == 6
        check_left_right_orthonormal(psi; atol=1e-10)

        move_center!(psi, 1)
        @test center(psi) == 1
        check_left_right_orthonormal(psi; atol=1e-10)
        @test isapprox(norm(psi), n_before; atol=1e-10)
    end
end

# ---------------------------------------------------------------------------
# make_phi / update_sites! round-trip
# ---------------------------------------------------------------------------

@testset "make_phi / update_sites! 1-site round-trip" begin
    for T in (Float64, ComplexF64)
        psi = make_random_mps(T, 6, 2, 4)
        orthogonalize!(psi); normalize!(psi)
        # Move center to 3 first
        move_center!(psi, 3)

        psi_ref = copy(psi)

        # 1-site sweep right at p=3
        phi = make_phi(psi, 3; n=1)
        @test size(phi) == size(psi[3])
        update_sites!(psi, 3, phi; absorb="right")
        @test center(psi) == 4

        # Norm preserved (no truncation)
        @test isapprox(norm(psi), 1.0; atol=1e-10)

        # Inner product with original ≈ 1 (same state)
        @test isapprox(abs(direct_inner(psi_ref, psi)), 1.0; atol=1e-10)
    end
end

@testset "make_phi / update_sites! 2-site round-trip" begin
    for T in (Float64, ComplexF64)
        psi = make_random_mps(T, 6, 2, 4)
        orthogonalize!(psi); normalize!(psi)
        move_center!(psi, 3)

        psi_ref = copy(psi)

        phi = make_phi(psi, 3; n=2)
        @test ndims(phi) == 4
        @test size(phi, 1) == size(psi[3], 1)
        @test size(phi, 2) == size(psi[3], 2)
        @test size(phi, 3) == size(psi[4], 2)
        @test size(phi, 4) == size(psi[4], 3)

        update_sites!(psi, 3, phi; absorb="right")
        @test center(psi) == 4
        @test isapprox(norm(psi), 1.0; atol=1e-10)
        @test isapprox(abs(direct_inner(psi_ref, psi)), 1.0; atol=1e-10)

        # Same again with absorb="left"
        psi2 = copy(psi_ref)
        phi2 = make_phi(psi2, 3; n=2)
        update_sites!(psi2, 3, phi2; absorb="left")
        @test center(psi2) == 3
        @test isapprox(norm(psi2), 1.0; atol=1e-10)
        @test isapprox(abs(direct_inner(psi_ref, psi2)), 1.0; atol=1e-10)
    end
end

# ---------------------------------------------------------------------------
# set_sites! atomicity and callback firing
# ---------------------------------------------------------------------------

@testset "callbacks" begin
    psi = make_random_mps(Float64, 4, 2, 3)
    orthogonalize!(psi); normalize!(psi)

    fired = Int[]
    register_callback!(psi, site -> push!(fired, site))

    move_center!(psi, 3)   # Each shift fires set_sites! → 2 sites at a time
    @test !isempty(fired)
end

# ---------------------------------------------------------------------------
# copy independence
# ---------------------------------------------------------------------------

@testset "copy" begin
    psi = make_random_mps(Float64, 4, 2, 3)
    orthogonalize!(psi); normalize!(psi)
    psi2 = copy(psi)

    move_center!(psi, 4)
    # psi2 should not be affected
    @test psi2.center_left == 1
    @test psi2.center_right == 1
    @test isapprox(norm(psi2), 1.0; atol=1e-10)
end
