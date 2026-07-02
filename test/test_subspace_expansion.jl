using Test
using LinearAlgebra
using TensorOperations

include("../src/Tensor/TensorUtils.jl")
include("../src/MPS/MPS.jl")
include("../src/MPS/MPO.jl")
include("../src/MPS/Init.jl")
include("../src/MPS/Operations.jl")
include("../src/DMRG/Environment.jl")
include("../src/DMRG/EffectiveOperators.jl")
include("../src/DMRG/SubspaceExpansion.jl")

# A random MPO with matching left/right boundary bonds of 1.
function se_random_mpo(::Type{T}, N::Int, d::Int, D::Int) where {T<:Number}
    md = [1; fill(D, N-1); 1]
    return MPO([randn(T, md[k], d, d, md[k+1]) for k in 1:N])
end

# Contract an MPS to its dense state vector (site 1 = slowest index), for the
# state-invariance checks below.  Independent of the code under test.
function dense_vector(psi::MPS{T}) where {T}
    v = ones(T, 1, 1)                    # (physical-so-far, right bond)
    for A in psi._tensors
        l, d, r = size(A)
        # v: (P, l); grow physical index by d.
        @tensor w[P, dd, r] := v[P, l] * A[l, dd, r]
        v = reshape(w, size(w, 1) * d, r)
    end
    return vec(v)                        # right boundary bond is 1
end

@testset "SubspaceExpansion" begin

    @testset "expansion_term shape ($T, $dir)" for T in (Float64, ComplexF64),
                                                    dir in ("right", "left")
        Dw = 4          # MPO bond
        A  = randn(T, 3, 2, 5)                 # (l, i, r)
        if dir == "right"
            env = randn(T, 3, Dw, 3)           # Lenv (dn, w, up); up = A's left bond
            W   = randn(T, Dw, 2, 2, Dw)       # (wl, ip, i, wr)
            P = expansion_term(env, A, W, dir)
            @test size(P) == (3, 2, 5 * Dw)    # right bond flattened with wr
        else
            env = randn(T, 5, Dw, 5)           # Renv (dn, w, up); up = A's right bond
            W   = randn(T, Dw, 2, 2, Dw)
            P = expansion_term(env, A, W, dir)
            @test size(P) == (3 * Dw, 2, 5)    # left bond flattened with wl
        end
    end

    # The core correctness guarantee: expansion widens the shared bond but the
    # represented state is UNCHANGED (the padding block of the neighbour is zero).
    @testset "state invariance, rightward ($T)" for T in (Float64, ComplexF64)
        N, d, D, Dw = 5, 2, 3, 3
        psi = random_mps(T, N, d, D)
        H   = se_random_mpo(T, N, d, Dw)
        p   = 3

        env = OperatorEnv(psi, psi, H; init_center=p)
        update_envs!(env, p, p)
        Lenv = getenv(env, p - 1)

        v_before = dense_vector(psi)
        Ap_e, Anext_e = expand_bond(psi[p], psi[p+1], Lenv, H[p], "right"; alpha=0.7)

        # widened shared bond
        @test size(Ap_e, 3) > size(psi[p], 3)
        @test size(Ap_e, 3) == size(Anext_e, 1)
        # left bond of Ap and right bond of Anext untouched
        @test size(Ap_e, 1) == size(psi[p], 1)
        @test size(Anext_e, 3) == size(psi[p+1], 3)

        psi2 = MPS([k == p ? Ap_e : k == p+1 ? Anext_e : copy(psi[k]) for k in 1:N])
        @test dense_vector(psi2) ≈ v_before
    end

    @testset "state invariance, leftward ($T)" for T in (Float64, ComplexF64)
        N, d, D, Dw = 5, 2, 3, 3
        psi = random_mps(T, N, d, D)
        H   = se_random_mpo(T, N, d, Dw)
        p   = 3                                    # site being expanded
        # neighbour is the LEFT one (p-1); grow the bond between them.

        env = OperatorEnv(psi, psi, H; init_center=p)
        update_envs!(env, p, p)
        Renv = getenv(env, p + 1)

        v_before = dense_vector(psi)
        Ap_e, Aprev_e = expand_bond(psi[p], psi[p-1], Renv, H[p], "left"; alpha=0.5)

        @test size(Ap_e, 1) > size(psi[p], 1)      # left bond of Ap widened
        @test size(Ap_e, 1) == size(Aprev_e, 3)    # matches right bond of neighbour
        @test size(Ap_e, 3) == size(psi[p], 3)     # right bond of Ap untouched
        @test size(Aprev_e, 1) == size(psi[p-1], 1)

        psi2 = MPS([k == p ? Ap_e : k == p-1 ? Aprev_e : copy(psi[k]) for k in 1:N])
        @test dense_vector(psi2) ≈ v_before
    end

    # alpha = 0 leaves the original tensors as the leading block (plus a zero
    # block): the widened Ap must equal the original padded with zeros.
    @testset "alpha=0 is a pure zero-pad ($T)" for T in (Float64, ComplexF64)
        N, d, D, Dw = 4, 2, 3, 2
        psi = random_mps(T, N, d, D)
        H   = se_random_mpo(T, N, d, Dw)
        p   = 2
        env = OperatorEnv(psi, psi, H; init_center=p); update_envs!(env, p, p)
        Lenv = getenv(env, p - 1)

        Ap_e, _ = expand_bond(psi[p], psi[p+1], Lenv, H[p], "right"; alpha=0.0)
        r0 = size(psi[p], 3)
        @test Ap_e[:, :, 1:r0] ≈ psi[p]            # leading block is the original
        @test all(iszero, Ap_e[:, :, r0+1:end])    # appended block is zero
    end

    # The enrichment directions are exactly H|psi>'s local action: the appended
    # block of Ap equals alpha * expansion_term (up to the direct-sum layout).
    @testset "enrichment block = alpha * H-action ($T)" for T in (Float64, ComplexF64)
        N, d, D, Dw = 5, 2, 3, 3
        psi = random_mps(T, N, d, D)
        H   = se_random_mpo(T, N, d, Dw)
        p   = 3
        env = OperatorEnv(psi, psi, H; init_center=p); update_envs!(env, p, p)
        Lenv = getenv(env, p - 1)

        alpha = 0.9
        P = expansion_term(Lenv, psi[p], H[p], "right")
        Ap_e, _ = expand_bond(psi[p], psi[p+1], Lenv, H[p], "right"; alpha=alpha)
        r0 = size(psi[p], 3)
        @test Ap_e[:, :, r0+1:end] ≈ T(alpha) .* P
    end

    @testset "bad direction errors" begin
        A   = randn(Float64, 2, 2, 2)
        env = randn(Float64, 2, 2, 2)
        W   = randn(Float64, 2, 2, 2, 2)
        @test_throws ErrorException expansion_term(env, A, W, "up")
    end
end
