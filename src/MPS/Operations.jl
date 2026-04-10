# Operations.jl
#
# High-level standalone operations on MPS / MPO objects:
#   inner(psi, phi)              -> <psi|phi>
#   expectation(psi, mpo, phi)   -> <psi|mpo|phi>
#   mps_sum(psi, phi)            -> MPS for |psi> + |phi>
#   mpo_sum(W1, W2)              -> MPO for W1 + W2
#   exact_apply_mpo(mpo, mps)    -> exact MPS for MPO|mps>
#   mpo_product(mpo1, mpo2)      -> MPO for mpo1 @ mpo2
#   fit_apply_mpo                -> stub (filled in after DMRG Environment)
#   fit_mpo_product              -> stub
#
# Conventions: bra = first MPS argument, always conjugated; ket = second.

"""
    inner(psi, phi) -> Number

Return `<psi|phi>` via direct left-to-right contraction.  `psi` is the bra
(conjugated) and `phi` is the ket.
"""
function inner(psi::MPS{T1}, phi::MPS{T2}) where {T1<:Number, T2<:Number}
    length(psi) == length(phi) ||
        error("MPS length mismatch: $(length(psi)) vs $(length(phi)).")
    T = promote_type(T1, T2)
    env = ones(T, 1, 1)   # (dn = ket virtual, up = bra virtual)
    for k in 1:length(phi)
        K = phi[k]
        B = psi[k]
        @tensor env_new[dn, up] := env[x, y] * K[x, i, dn] * conj(B)[y, i, up]
        env = env_new
    end
    return only(env)
end

"""
    expectation(psi, mpo, phi) -> Number

Return `<psi|mpo|phi>` via direct left-to-right contraction.
"""
function expectation(psi::MPS{T1}, mpo::MPO{T2}, phi::MPS{T3}) where {T1,T2,T3}
    N = length(psi)
    (length(phi) == N && length(mpo) == N) ||
        error("Length mismatch: psi=$N, mpo=$(length(mpo)), phi=$(length(phi)).")
    T = promote_type(T1, T2, T3)
    env = ones(T, 1, 1, 1)   # (dn = ket virtual, w = MPO virtual, up = bra virtual)
    for k in 1:N
        K = phi[k]
        W = mpo[k]
        B = psi[k]
        @tensor env_new[dn, w, up] := env[x, m, y] *
                                      K[x, i, dn] *
                                      W[m, ip, i, w] *
                                      conj(B)[y, ip, up]
        env = env_new
    end
    return only(env)
end

# ---------------------------------------------------------------------------
# Sums via direct sum of virtual bonds
# ---------------------------------------------------------------------------

"""
    mps_sum(psi, phi) -> MPS

Return the MPS representing `|psi> + |phi>` via direct sum of virtual bonds.

- site 1   : direct-sum on dim 3 (r) only (left boundary bond is shared 1).
- interior : direct-sum on dims 1 (l) and 3 (r).
- site N   : direct-sum on dim 1 (l) only.
"""
function mps_sum(psi::MPS{T}, phi::MPS{T}) where {T}
    N = length(psi)
    N == length(phi) || error("MPS length mismatch: $N vs $(length(phi)).")
    N >= 2 || error("mps_sum requires at least 2 sites.")
    tensors = Vector{Array{T,3}}(undef, N)
    for k in 1:N
        A, B = psi[k], phi[k]
        if k == 1
            tensors[k] = direct_sum(A, B, [3])
        elseif k == N
            tensors[k] = direct_sum(A, B, [1])
        else
            tensors[k] = direct_sum(A, B, [1, 3])
        end
    end
    return MPS(tensors)
end

"""
    mpo_sum(W1, W2) -> MPO

Return the MPO representing `W1 + W2` via direct sum of virtual bonds.
"""
function mpo_sum(W1::MPO{T}, W2::MPO{T}) where {T}
    N = length(W1)
    N == length(W2) || error("MPO length mismatch: $N vs $(length(W2)).")
    N >= 2 || error("mpo_sum requires at least 2 sites.")
    tensors = Vector{Array{T,4}}(undef, N)
    for k in 1:N
        A, B = W1[k], W2[k]
        if k == 1
            tensors[k] = direct_sum(A, B, [4])
        elseif k == N
            tensors[k] = direct_sum(A, B, [1])
        else
            tensors[k] = direct_sum(A, B, [1, 4])
        end
    end
    return MPO(tensors)
end

# ---------------------------------------------------------------------------
# Apply MPO to MPS / multiply MPOs (exact, no truncation)
# ---------------------------------------------------------------------------

"""
    exact_apply_mpo(mpo, mps) -> MPS

Return the exact (un-truncated) MPS representing `MPO|mps>`.
Bond dimensions become the product `D_mpo[k] * D_mps[k]`.
The new physical leg comes from the MPO's `ip` leg.
"""
function exact_apply_mpo(mpo::MPO{T1}, mps::MPS{T2}) where {T1, T2}
    N = length(mpo)
    N == length(mps) || error("Length mismatch: mpo=$N, mps=$(length(mps)).")
    T = promote_type(T1, T2)
    tensors = Vector{Array{T,3}}(undef, N)
    for k in 1:N
        W = mpo[k]
        A = mps[k]
        # Contract MPO physical i with MPS physical i:
        #   W[ml, ip, i, mr] * A[al, i, ar]  →  T[ml, al, ip, mr, ar]
        @tensor TT[ml, al, ip, mr, ar] := W[ml, ip, i, mr] * A[al, i, ar]
        ml, al, ip, mr, ar = size(TT)
        tensors[k] = reshape(TT, (ml*al, ip, mr*ar))
    end
    return MPS(tensors)
end

"""
    mpo_product(mpo1, mpo2) -> MPO

Return the MPO representing `mpo1 @ mpo2` (mpo1 applied after mpo2):
contracts `mpo1.i` with `mpo2.ip`.  Bond dimensions multiply.
"""
function mpo_product(mpo1::MPO{T1}, mpo2::MPO{T2}) where {T1, T2}
    N = length(mpo1)
    N == length(mpo2) || error("MPO length mismatch: $N vs $(length(mpo2)).")
    T = promote_type(T1, T2)
    tensors = Vector{Array{T,4}}(undef, N)
    for k in 1:N
        U = mpo1[k]
        L = mpo2[k]
        # U[ul, ip, m, ur] * L[ll, m, i, lr]  →  T[ul, ll, ip, i, ur, lr]
        @tensor TT[ul, ll, ip, i, ur, lr] := U[ul, ip, m, ur] * L[ll, m, i, lr]
        ul, ll, ip, i, ur, lr = size(TT)
        tensors[k] = reshape(TT, (ul*ll, ip, i, ur*lr))
    end
    return MPO(tensors)
end

# ---------------------------------------------------------------------------
# fit_apply_mpo
# ---------------------------------------------------------------------------
#
# Variationally fit `|fitmps⟩ ≈ MPO|mps_input⟩` by sweeping over the sites of
# `fitmps` and solving each local subspace problem exactly via one application
# of the effective operator (no eigensolver).
#
# Requires DMRG/Environment.jl and DMRG/EffectiveOperators.jl to be loaded
# (the package module includes them before this file at the package level —
# in tests, include them before this method is called).

"""
    fit_apply_mpo(mpo, mps_input, fitmps; num_center=2, nsweep=1,
                  max_dim=nothing, cutoff=0.0, normalize=false) -> MPS

Fit `fitmps ≈ mpo|mps_input⟩` by sweeping.  `fitmps` is modified in-place
and also returned.  `fitmps` must have center == 1 at entry.
"""
function fit_apply_mpo(mpo::MPO, mps_input::MPS, fitmps::MPS;
                       num_center::Int=2,
                       nsweep::Int=1,
                       max_dim::Union{Int,Nothing}=nothing,
                       cutoff::Real=0.0,
                       normalize::Bool=false)
    N = length(mps_input)
    (length(mpo) == N && length(fitmps) == N) ||
        error("Length mismatch: mpo=$(length(mpo)), mps_input=$N, fitmps=$(length(fitmps)).")
    num_center in (1, 2) ||
        error("num_center must be 1 or 2; got $num_center.")
    center(fitmps) == 1 ||
        error("fitmps.center must be 1 at entry; got $(center(fitmps)).")

    n = num_center
    # Build environment for <fitmps|MPO|mps_input>: bra=fitmps, ket=mps_input
    op_env = OperatorEnv(mps_input, fitmps, mpo; init_center=1)

    function _local!(p::Int, absorb::String)
        update_envs!(op_env, p, p + n - 1)
        Lenv = getenv(op_env, p - 1)
        Renv = getenv(op_env, p + n)
        mpo_tensors = ntuple(k -> mpo[p + k - 1], n)
        effH = EffOperator(Lenv, Renv, mpo_tensors...)
        phi_in  = make_phi(mps_input, p; n=n)
        phi_new = apply(effH, phi_in)
        update_sites!(fitmps, p, phi_new;
                      max_dim=max_dim, cutoff=cutoff, absorb=absorb)
    end

    for _ in 1:nsweep
        # Sweep right: p = 1 .. N-1 (both 1-site and 2-site)
        for p in 1:N-1
            _local!(p, "right")
        end
        # Sweep left
        # 2-site: p = N-1 .. 1 ; 1-site: p = N .. 2
        left_start = N - n + 1
        left_stop  = n == 2 ? 1 : 2
        for p in left_start:-1:left_stop
            _local!(p, "left")
        end
    end

    if normalize
        LinearAlgebra.normalize!(fitmps)
    end
    return fitmps
end
