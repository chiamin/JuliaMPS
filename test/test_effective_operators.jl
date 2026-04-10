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

function make_random_mpo(::Type{T}, N::Int, d::Int, D::Int) where {T<:Number}
    md = [1; fill(D, N-1); 1]
    tensors = [randn(T, md[k], d, d, md[k+1]) for k in 1:N]
    return MPO(tensors)
end

@testset "EffOperator 1-site shape and energy" begin
    for T in (Float64, ComplexF64)
        N = 5
        psi = random_mps(T, N, 2, 3)
        W   = make_random_mpo(T, N, 2, 3)

        # Make psi Hermitian-friendly: just use it as is, eigvalue test below
        # uses <phi|H_eff|phi> compared to <psi|H|psi>.

        env = OperatorEnv(psi, psi, W; init_center=3)
        update_envs!(env, 3, 3)

        L = getenv(env, 2)
        R = getenv(env, 4)
        effH = EffOperator(L, R, W[3])

        phi = make_phi(psi, 3; n=1)
        out = apply(effH, phi)
        @test size(out) == size(phi)

        # <phi|H_eff|phi> should equal <psi|W|psi> when phi = psi[3] in canonical form
        e_eff = sum(conj.(phi) .* out)
        e_full = expectation(psi, W, psi)
        @test isapprox(e_eff, e_full; atol=1e-10)
    end
end

@testset "EffOperator 2-site shape and energy" begin
    for T in (Float64, ComplexF64)
        N = 6
        psi = random_mps(T, N, 2, 3)
        W   = make_random_mpo(T, N, 2, 3)

        env = OperatorEnv(psi, psi, W; init_center=3)
        update_envs!(env, 3, 4)

        L = getenv(env, 2)
        R = getenv(env, 5)
        effH = EffOperator(L, R, W[3], W[4])

        phi = make_phi(psi, 3; n=2)
        out = apply(effH, phi)
        @test size(out) == size(phi)
        e_eff  = sum(conj.(phi) .* out)
        e_full = expectation(psi, W, psi)
        @test isapprox(e_eff, e_full; atol=1e-10)
    end
end

@testset "EffOperator 0-site (bond tensor)" begin
    T = Float64
    N = 5
    psi = random_mps(T, N, 2, 3)
    W   = make_random_mpo(T, N, 2, 3)

    env = OperatorEnv(psi, psi, W; init_center=3)
    # 0-site at "bond between sites 3 and 4": need L=LR[3], R=LR[4]
    # That requires both 3 and 4 are valid (stale window must not contain them)
    update_envs!(env, 1, 1)   # makes LR[2..N+1] valid
    update_envs!(env, N, N)   # makes LR[0..N-1] valid; now stale=[N,N]
    # But we need LR[3] (left env) AND LR[4] (right env). LR[3] is valid as a
    # left env when stale > 3; LR[4] is valid as a right env when stale < 4.
    update_envs!(env, 4, 4)   # stale=[4,4]; LR[3] left, LR[5] right
    L = getenv(env, 3)
    R = getenv(env, 5)
    # Bond tensor between sites 4 — actually 0-site at "bond p" with no MPO
    # tensor uses W[p] absent. In 0-site DMRG/TDVP, the bond tensor sits
    # between two adjacent sites and you contract L and R with the MPO bond
    # absent. Let's just check shape and contraction succeeds.
    effH = EffOperator(L, R)
    # phi shape (l, r): l = bond dim between site 3 and 4
    Dl = size(L, 1)   # ket virtual = MPS bond
    Dr = size(R, 1)
    phi0 = randn(T, Dl, Dr)
    out0 = apply(effH, phi0)
    @test size(out0) == size(phi0)
end

@testset "EffVector inner product" begin
    for T in (Float64, ComplexF64)
        N = 5
        psi   = random_mps(T, N, 2, 3)
        ortho = random_mps(T, N, 2, 3)

        ve = VectorEnv(ortho, psi; init_center=3)
        update_envs!(ve, 3, 3)
        L = getenv(ve, 2)
        R = getenv(ve, 4)
        eff_vec = EffVector(L, R, ortho[3])

        phi = make_phi(psi, 3; n=1)
        ov = inner(eff_vec, phi)
        # eff_vec.tensor is the projected ortho (ket-form);
        # inner(eff_vec, phi) = <ortho|psi> (local) = <ortho|psi> (global)
        # = inner(ortho, psi) by our convention (bra is conjugated).
        @test isapprox(ov, inner(ortho, psi); atol=1e-10)
    end
end

@testset "EffOperator with penalty term" begin
    T = Float64
    N = 5
    psi   = random_mps(T, N, 2, 3)
    ortho = random_mps(T, N, 2, 3)
    W     = make_random_mpo(T, N, 2, 3)

    op_env = OperatorEnv(psi, psi, W; init_center=3)
    ve     = VectorEnv(ortho, psi; init_center=3)
    update_envs!(op_env, 3, 3)
    update_envs!(ve, 3, 3)

    L = getenv(op_env, 2); R = getenv(op_env, 4)
    Lv = getenv(ve, 2);    Rv = getenv(ve, 4)

    effH    = EffOperator(L, R, W[3])
    eff_vec = EffVector(Lv, Rv, ortho[3])
    add_term!(effH, eff_vec, 100.0)

    phi = make_phi(psi, 3; n=1)
    out = apply(effH, phi)

    # Compare to manual: H_eff |phi⟩ + 100 * <ortho|psi> * |Φ_0⟩
    out_h = let
        @tensor tmp[lo, ip, ro] :=
            L[ldn, wL, lo] * phi[ldn, i, rdn] *
            W[3][wL, ip, i, wR] * R[rdn, wR, ro]
        tmp
    end
    expected = out_h .+ 100.0 .* inner(ortho, psi) .* eff_vec.tensor
    @test isapprox(out, expected; atol=1e-10)
end
