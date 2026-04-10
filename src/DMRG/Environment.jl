# Environment.jl
#
# Left/right environment tensor cache for DMRG and related algorithms.
#
# Indexing convention (Julia 1-indexed sites):
#
#   LR[0]      : left boundary (no sites absorbed)
#   LR[k] for k=1..N : env that has absorbed site k (interpreted as either
#                      "left env covering sites 1..k" or
#                      "right env covering sites k..N", depending on the
#                      stale-window position)
#   LR[N+1]    : right boundary (no sites absorbed)
#
# Stale window [centerL, centerR] (both in [1, N]):
#   Valid left envs  : LR[0], LR[1], ..., LR[centerL - 1]
#   Valid right envs : LR[centerR + 1], ..., LR[N], LR[N+1]
#   Stale            : LR[centerL], ..., LR[centerR]
#
# Tensor leg orders:
#   OperatorEnv tensor : Array{T,3}, dims = (dn, w, up)
#       dn = ket virtual bond, w = MPO virtual, up = bra virtual
#   VectorEnv tensor   : Array{T,2}, dims = (dn, up)
#
# Boundary tensors are scalar (1×1×1 or 1×1) of value 1.
#
# Callbacks: each env subclass registers a closure on the MPS/MPO it watches
# so that delete!(env, site) is called automatically on tensor updates.

abstract type AbstractLREnv end

# ---------------------------------------------------------------------------
# OperatorEnv  <bra| MPO |ket>
# ---------------------------------------------------------------------------

mutable struct OperatorEnv{T<:Number} <: AbstractLREnv
    N       :: Int
    centerL :: Int
    centerR :: Int
    LR      :: Dict{Int, Array{T,3}}
    mps1    :: MPS    # ket
    mps2    :: MPS    # bra
    mpo     :: MPO
end

"""
    OperatorEnv(mps1, mps2, mpo; init_center=1) -> OperatorEnv

Build environment tensors for `<mps2|mpo|mps1>`.  After construction, all
environments are valid except the stale window `[init_center, init_center]`.

`mps1` is the ket, `mps2` the bra (use the same MPS object for ground-state DMRG).
"""
function OperatorEnv(mps1::MPS, mps2::MPS, mpo::MPO; init_center::Int=1)
    N = length(mps1)
    (length(mps2) == N && length(mpo) == N) ||
        error("Length mismatch: mps1=$N, mps2=$(length(mps2)), mpo=$(length(mpo)).")
    1 <= init_center <= N ||
        error("init_center=$init_center out of range [1, $N].")
    T = promote_type(eltype(mps1), eltype(mps2), eltype(mpo))
    LR = Dict{Int, Array{T,3}}()
    LR[0]   = ones(T, 1, 1, 1)
    LR[N+1] = ones(T, 1, 1, 1)
    env = OperatorEnv{T}(N, 1, N, LR, mps1, mps2, mpo)

    # Register callbacks: invalidate the env when any input changes.
    cb = site -> delete!(env, site)
    register_callback!(mps1, cb)
    if mps2 !== mps1
        register_callback!(mps2, cb)
    end
    register_callback!(mpo, cb)

    update_envs!(env, init_center, init_center)
    return env
end

# ---------------------------------------------------------------------------
# Common stale-window helpers
# ---------------------------------------------------------------------------

"""
    delete!(env, site)

Mark `LR[site]` as stale by expanding the stale window to include `site`.
Called automatically when an underlying MPS/MPO tensor is updated.
"""
function Base.delete!(env::AbstractLREnv, site::Int)
    if 1 <= site <= env.N
        env.centerL = min(env.centerL, site)
        env.centerR = max(env.centerR, site)
    end
    return env
end

"""
    update_envs!(env, centerL, centerR=centerL)

Compute missing environments to shrink the stale window to `[centerL, centerR]`.
"""
function update_envs!(env::AbstractLREnv, centerL::Int, centerR::Int=centerL)
    centerL <= centerR + 1 ||
        error("centerL=$centerL > centerR+1=$(centerR+1): invalid stale window.")
    1 <= centerL && centerR <= env.N ||
        error("Window [$centerL, $centerR] out of range [1, $(env.N)].")

    # Grow left envs: from old centerL up to new centerL-1.
    # LR[k] depends on LR[k-1] which must already be valid (k-1 < old centerL).
    for k in env.centerL : (centerL - 1)
        env.LR[k] = _grow_left(env, k, env.LR[k-1])
    end
    # Grow right envs: from old centerR down to new centerR+1.
    for k in env.centerR : -1 : (centerR + 1)
        env.LR[k] = _grow_right(env, k, env.LR[k+1])
    end
    env.centerL = centerL
    env.centerR = centerR
    return env
end

"""
    getenv(env, k) -> Array

Return the validated environment at index `k`.  Errors if `k` is in the stale
window.
"""
function getenv(env::AbstractLREnv, k::Int)
    (0 <= k <= env.N + 1) ||
        error("Environment index $k out of range [0, $(env.N + 1)].")
    if env.centerL <= k <= env.centerR
        error("LR[$k] is in the stale window [$(env.centerL), $(env.centerR)]; " *
              "call update_envs! first.")
    end
    return env.LR[k]
end

# ---------------------------------------------------------------------------
# OperatorEnv contractions
# ---------------------------------------------------------------------------

"""
    grow_left_op(prev_env, ket, mpo, bra) -> Array{T,3}

Absorb one site (from the left) into `prev_env` using explicit tensors.
Used both by `OperatorEnv._grow_left` and by TDVP's 0-site env construction.
"""
function grow_left_op(prev_env::Array{T,3}, ket::Array{T,3},
                      mpo::Array{T,4}, bra::Array{T,3}) where {T}
    @tensor new_env[dnR, wR, upR] :=
        prev_env[dnL, wL, upL] *
        ket[dnL, i, dnR] *
        mpo[wL, ip, i, wR] *
        conj(bra)[upL, ip, upR]
    return new_env
end

"""
    grow_right_op(next_env, ket, mpo, bra) -> Array{T,3}

Absorb one site (from the right) into `next_env` using explicit tensors.
"""
function grow_right_op(next_env::Array{T,3}, ket::Array{T,3},
                       mpo::Array{T,4}, bra::Array{T,3}) where {T}
    @tensor new_env[dnL, wL, upL] :=
        next_env[dnR, wR, upR] *
        ket[dnL, i, dnR] *
        mpo[wL, ip, i, wR] *
        conj(bra)[upL, ip, upR]
    return new_env
end

_grow_left(env::OperatorEnv{T}, p::Int, prev_env::Array{T,3}) where {T} =
    grow_left_op(prev_env, env.mps1[p], env.mpo[p], env.mps2[p])

_grow_right(env::OperatorEnv{T}, p::Int, next_env::Array{T,3}) where {T} =
    grow_right_op(next_env, env.mps1[p], env.mpo[p], env.mps2[p])

# ---------------------------------------------------------------------------
# VectorEnv  <mps2|mps1>  (used for excited-state penalty)
# ---------------------------------------------------------------------------

mutable struct VectorEnv{T<:Number} <: AbstractLREnv
    N       :: Int
    centerL :: Int
    centerR :: Int
    LR      :: Dict{Int, Array{T,2}}
    mps1    :: MPS    # ket
    mps2    :: MPS    # bra
end

"""
    VectorEnv(mps1, mps2; init_center=1) -> VectorEnv

Build overlap environments for `<mps2|mps1>`.
"""
function VectorEnv(mps1::MPS, mps2::MPS; init_center::Int=1)
    N = length(mps1)
    length(mps2) == N || error("Length mismatch: $N vs $(length(mps2)).")
    1 <= init_center <= N ||
        error("init_center=$init_center out of range [1, $N].")
    T = promote_type(eltype(mps1), eltype(mps2))
    LR = Dict{Int, Array{T,2}}()
    LR[0]   = ones(T, 1, 1)
    LR[N+1] = ones(T, 1, 1)
    env = VectorEnv{T}(N, 1, N, LR, mps1, mps2)
    cb = site -> delete!(env, site)
    register_callback!(mps1, cb)
    if mps2 !== mps1
        register_callback!(mps2, cb)
    end
    update_envs!(env, init_center, init_center)
    return env
end

function _grow_left(env::VectorEnv{T}, p::Int, prev_env::Array{T,2}) where {T}
    ket = env.mps1[p]
    bra = env.mps2[p]
    @tensor new_env[dnR, upR] :=
        prev_env[dnL, upL] * ket[dnL, i, dnR] * conj(bra)[upL, i, upR]
    return new_env
end

function _grow_right(env::VectorEnv{T}, p::Int, next_env::Array{T,2}) where {T}
    ket = env.mps1[p]
    bra = env.mps2[p]
    @tensor new_env[dnL, upL] :=
        next_env[dnR, upR] * ket[dnL, i, dnR] * conj(bra)[upL, i, upR]
    return new_env
end
