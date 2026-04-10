using Test

include("../src/MPS/MPO.jl")

# Build a trivial random MPO
function make_random_mpo(::Type{T}, N::Int, d::Int, D::Int) where {T<:Number}
    md = [1; fill(D, N-1); 1]
    tensors = [randn(T, md[k], d, d, md[k+1]) for k in 1:N]
    return MPO(tensors)
end

@testset "MPO construction" begin
    W = make_random_mpo(Float64, 5, 2, 3)
    @test length(W) == 5
    @test eltype(W) == Float64
    @test phys_dims(W) == [2,2,2,2,2]
    @test mpo_dims(W) == [1,3,3,3,3,1]

    # Boundary
    @test size(W[1], 1) == 1
    @test size(W[end], 4) == 1

    # Physical match at every site
    for k in 1:length(W)
        @test size(W[k], 2) == size(W[k], 3)
    end
end

@testset "MPO validation errors" begin
    # ip != i (physical legs mismatch)
    bad = [randn(1, 2, 3, 1)]
    @test_throws ErrorException MPO(bad)

    # Boundary != 1
    bad2 = [randn(2, 2, 2, 1)]
    @test_throws ErrorException MPO(bad2)

    # Neighbour mismatch
    bad3 = [randn(1, 2, 2, 3), randn(2, 2, 2, 1)]
    @test_throws ErrorException MPO(bad3)
end

@testset "MPO copy and setindex" begin
    W = make_random_mpo(Float64, 4, 2, 2)
    W2 = copy(W)
    # Modify W2 in place
    new_site = randn(2, 2, 2, 2)
    W2[2] = new_site
    @test W2[2] == new_site
    @test W[2] != new_site   # Original unaffected

    # Bad setindex throws
    @test_throws ErrorException (W2[2] = randn(2, 3, 2, 2))   # ip != i
end
