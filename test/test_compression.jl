using Test
using LinearAlgebra
using TensorOperations

include("../src/Tensor/TensorUtils.jl")
include("../src/MPS/MPS.jl")
include("../src/MPS/MPO.jl")
include("../src/MPS/Init.jl")
include("../src/MPS/Operations.jl")
include("../src/MPS/Compression.jl")
include("../src/MPS/MPOCompression.jl")

function make_random_mpo(::Type{T}, N::Int, d::Int, D::Int) where {T<:Number}
    md = [1; fill(D, N-1); 1]
    tensors = [randn(T, md[k], d, d, md[k+1]) for k in 1:N]
    return MPO(tensors)
end

# ---------------------------------------------------------------------------
# svd_compress_mps
# ---------------------------------------------------------------------------

@testset "svd_compress_mps without truncation" begin
    for T in (Float64, ComplexF64)
        psi = random_mps(T, 6, 2, 4)
        phi = svd_compress_mps(psi)   # no truncation
        # State should be (almost) unchanged
        @test isapprox(abs(inner(psi, phi)), 1.0; atol=1e-10)
        @test isapprox(norm(phi), 1.0; atol=1e-10)
    end
end

@testset "svd_compress_mps with truncation" begin
    psi = random_mps(Float64, 6, 2, 8)
    phi = svd_compress_mps(psi; max_dim=4)
    @test maximum(bond_dims(phi)) <= 4
    # Compressed state still has unit-ish norm (norm preserved by SVD up to discarded weight)
    @test norm(phi) <= 1.0 + 1e-10
end

# ---------------------------------------------------------------------------
# svd_compress_mpo
# ---------------------------------------------------------------------------

@testset "svd_compress_mpo without truncation" begin
    for T in (Float64, ComplexF64)
        W = make_random_mpo(T, 5, 2, 3)
        Wc = svd_compress_mpo(W)
        # Apply both to a random state and compare expectation values
        psi = random_mps(T, 5, 2, 3; normalize=false)
        phi = random_mps(T, 5, 2, 3; normalize=false)
        @test isapprox(expectation(psi, W, phi),
                       expectation(psi, Wc, phi); rtol=1e-10)
    end
end

@testset "svd_compress_mpo with truncation" begin
    W = make_random_mpo(Float64, 5, 2, 6)
    Wc = svd_compress_mpo(W; max_dim=3)
    @test maximum(mpo_dims(Wc)) <= 3
end
