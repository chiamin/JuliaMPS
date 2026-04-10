using Test
using LinearAlgebra

include("../src/Tensor/TensorUtils.jl")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Reconstruct full matrix from SVD output (absorb="right")
function reconstruct_svd_right(U, Vt, row_dims, col_dims)
    chi = size(U)[end]
    U_mat  = reshape(U,  prod(row_dims), chi)
    Vt_mat = reshape(Vt, chi, prod(col_dims))
    return reshape(U_mat * Vt_mat, (row_dims..., col_dims...))
end

# Reconstruct full matrix from SVD output (absorb=nothing)
function reconstruct_svd_none(U, s, Vt, row_dims, col_dims)
    chi = length(s)
    U_mat  = reshape(U,  prod(row_dims), chi)
    Vt_mat = reshape(Vt, chi, prod(col_dims))
    return reshape(U_mat * diagm(s) * Vt_mat, (row_dims..., col_dims...))
end

# ---------------------------------------------------------------------------
# svd_split: round-trip
# ---------------------------------------------------------------------------

@testset "svd_split round-trip" begin
    @testset "real, rank-2, n_row_legs=1" begin
        A = randn(4, 5)
        U, Vt, disc = svd_split(A, 1)
        @test size(U)  == (4, 4)
        @test size(Vt) == (4, 5)
        @test isapprox(reconstruct_svd_right(U, Vt, (4,), (5,)), A; atol=1e-12)
        @test disc ≈ 0.0 atol=1e-12
    end

    @testset "complex, rank-3, n_row_legs=2" begin
        A = randn(ComplexF64, 3, 4, 5)
        U, Vt, disc = svd_split(A, 2)
        chi = min(3*4, 5)   # = 5
        @test size(U)  == (3, 4, chi)
        @test size(Vt) == (chi, 5)
        @test isapprox(reconstruct_svd_right(U, Vt, (3,4), (5,)), A; atol=1e-12)
        @test disc ≈ 0.0 atol=1e-12
    end

    @testset "real, rank-3 MPS shape (l,i,r), n_row_legs=2" begin
        A = randn(4, 2, 4)
        U, Vt, disc = svd_split(A, 2)
        @test size(U, 3) <= 8   # chi <= min(l*i, r)
        @test isapprox(reconstruct_svd_right(U, Vt, (4,2), (4,)), A; atol=1e-12)
    end
end

# ---------------------------------------------------------------------------
# svd_split: absorb variants
# ---------------------------------------------------------------------------

@testset "svd_split absorb" begin
    A = randn(3, 5)

    @testset "absorb=right (default)" begin
        U, Vt, _ = svd_split(A, 1, absorb="right")
        # U should be isometry: U†U ≈ I
        U_mat = reshape(U, 3, size(U, 2))
        @test isapprox(U_mat' * U_mat, I; atol=1e-12)
        @test isapprox(reconstruct_svd_right(U, Vt, (3,), (5,)), A; atol=1e-12)
    end

    @testset "absorb=left" begin
        U, Vt, _ = svd_split(A, 1, absorb="left")
        # Vt should be isometry: Vt * Vt† ≈ I
        chi = size(Vt, 1)
        Vt_mat = reshape(Vt, chi, 5)
        @test isapprox(Vt_mat * Vt_mat', I; atol=1e-12)
        @test isapprox(reconstruct_svd_right(U, Vt, (3,), (5,)), A; atol=1e-12)
    end

    @testset "absorb=nothing" begin
        U, s, Vt, _ = svd_split(A, 1, absorb=nothing)
        @test ndims(s) == 1
        @test isapprox(reconstruct_svd_none(U, s, Vt, (3,), (5,)), A; atol=1e-12)
        # Both U and Vt are isometries when s is separate
        chi = length(s)
        U_mat  = reshape(U,  3, chi)
        Vt_mat = reshape(Vt, chi, 5)
        @test isapprox(U_mat' * U_mat, I; atol=1e-12)
        @test isapprox(Vt_mat * Vt_mat', I; atol=1e-12)
    end
end

# ---------------------------------------------------------------------------
# svd_split: maxdim truncation
# ---------------------------------------------------------------------------

@testset "svd_split maxdim" begin
    A = randn(6, 6)
    for maxdim in [1, 3, 5]
        U, Vt, disc = svd_split(A, 1; maxdim=maxdim)
        @test size(U, 2) == maxdim
        @test size(Vt, 1) == maxdim
        @test disc >= 0.0
        @test disc <= 1.0 + 1e-12
    end
end

# ---------------------------------------------------------------------------
# svd_split: cutoff truncation
# ---------------------------------------------------------------------------

@testset "svd_split cutoff" begin
    # Build a rank-2 matrix: only 2 non-zero singular values
    U0, _, V0 = svd(randn(4, 4))
    s0 = [10.0, 1.0, 0.0, 0.0]
    A = U0 * diagm(s0) * V0'

    # cutoff=0: keep all (but zero singular values may still be dropped due to float)
    U, Vt, disc = svd_split(A, 1; cutoff=0.0)
    @test disc ≈ 0.0 atol=1e-10

    # λ_2 = 1²/101 ≈ 0.0099; use cutoff=0.02 to drop it
    U, Vt, disc = svd_split(A, 1; cutoff=0.02)
    chi = size(U, 2)
    @test chi == 1
    # discarded ≈ λ_2 = 1/101
    @test isapprox(disc, 1.0^2 / (10.0^2 + 1.0^2); atol=1e-6)
end

# ---------------------------------------------------------------------------
# svd_split: complex values — test that conj is handled correctly
# ---------------------------------------------------------------------------

@testset "svd_split complex correctness" begin
    # U should satisfy U†U = I (not U^T U), which requires complex conj
    A = randn(ComplexF64, 4, 5)
    U, Vt, _ = svd_split(A, 1, absorb="right")
    chi = size(U, 2)
    U_mat = reshape(U, 4, chi)
    @test isapprox(U_mat' * U_mat, I(chi); atol=1e-12)
    @test isapprox(reconstruct_svd_right(U, Vt, (4,), (5,)), A; atol=1e-12)
end

# ---------------------------------------------------------------------------
# qr_split: round-trip
# ---------------------------------------------------------------------------

@testset "qr_split round-trip" begin
    @testset "real, rank-2" begin
        A = randn(4, 5)
        Q, R = qr_split(A, 1)
        chi = size(Q, 2)
        @test size(Q) == (4, chi)
        @test size(R) == (chi, 5)
        Q_mat = reshape(Q, 4, chi)
        @test isapprox(Q_mat' * Q_mat, I(chi); atol=1e-12)
        @test isapprox(reshape(Q_mat * reshape(R, chi, 5), 4, 5), A; atol=1e-12)
    end

    @testset "complex, rank-3, n_row_legs=2" begin
        A = randn(ComplexF64, 3, 4, 5)
        Q, R = qr_split(A, 2)
        chi = size(Q, 3)
        Q_mat = reshape(Q, 12, chi)
        @test isapprox(Q_mat' * Q_mat, I(chi); atol=1e-12)
        A_rec = reshape(Q_mat * reshape(R, chi, 5), 3, 4, 5)
        @test isapprox(A_rec, A; atol=1e-12)
    end
end

# ---------------------------------------------------------------------------
# direct_sum
# ---------------------------------------------------------------------------

@testset "direct_sum" begin
    @testset "rank-3, sum along dim 1 and 3 (MPS interior)" begin
        A = randn(2, 3, 4)
        B = randn(5, 3, 6)
        C = direct_sum(A, B, [1, 3])

        @test size(C) == (7, 3, 10)

        # A block
        @test isapprox(C[1:2, :, 1:4], A; atol=1e-14)
        # B block
        @test isapprox(C[3:7, :, 5:10], B; atol=1e-14)
        # off-diagonal blocks are zero
        @test all(iszero, C[1:2, :, 5:10])
        @test all(iszero, C[3:7, :, 1:4])
    end

    @testset "rank-3, sum along dim 3 only (MPS boundary)" begin
        A = randn(1, 3, 2)
        B = randn(1, 3, 4)
        C = direct_sum(A, B, [3])

        @test size(C) == (1, 3, 6)
        @test isapprox(C[:, :, 1:2], A; atol=1e-14)
        @test isapprox(C[:, :, 3:6], B; atol=1e-14)
    end

    @testset "rank-4, sum along dim 1 and 4 (MPO interior)" begin
        A = randn(2, 3, 3, 4)
        B = randn(5, 3, 3, 6)
        C = direct_sum(A, B, [1, 4])

        @test size(C) == (7, 3, 3, 10)
        @test isapprox(C[1:2, :, :, 1:4], A; atol=1e-14)
        @test isapprox(C[3:7, :, :, 5:10], B; atol=1e-14)
        @test all(iszero, C[1:2, :, :, 5:10])
        @test all(iszero, C[3:7, :, :, 1:4])
    end

    @testset "complex arrays" begin
        A = randn(ComplexF64, 2, 3, 4)
        B = randn(ComplexF64, 5, 3, 6)
        C = direct_sum(A, B, [1, 3])
        @test eltype(C) == ComplexF64
        @test isapprox(C[1:2, :, 1:4], A; atol=1e-14)
        @test isapprox(C[3:7, :, 5:10], B; atol=1e-14)
    end

    @testset "mixed real+complex promotes type" begin
        A = randn(Float64,    2, 3, 4)
        B = randn(ComplexF64, 2, 3, 4)
        C = direct_sum(A, B, [1])
        @test eltype(C) == ComplexF64
    end
end
