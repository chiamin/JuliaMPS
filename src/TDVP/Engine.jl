# TDVP/Engine.jl
#
# Time-dependent variational principle (TDVP) sweep engine for MPS.
#
# Time convention
# ---------------
# One full sweep (right then left) applies  exp(-dt * H)  to psi.
#   Forward step at each site/bond : exp(-dt/2 * H_eff)
#   Backward step at each bond/site: exp(+dt/2 * H_eff_0site or H_eff_1site)
#
# For real-time evolution by Δt    : pass dt = 1im * Δt
# For imaginary-time by Δτ         : pass dt = Δτ  (real)
#
# Usage
# -----
#     engine = TDVPEngine(psi, H)
#     for _ in 1:n_steps
#         sweep!(engine, dt; num_center=1)   # or num_center=2
#     end
#
# `psi` is modified in-place.  Environments are built once at construction.

mutable struct TDVPEngine{T<:Number}
    psi    :: MPS{T}
    H      :: MPO{T}
    op_env :: OperatorEnv{T}
end

"""
    TDVPEngine(psi, H) -> TDVPEngine

Build the engine.  `psi` must already be canonicalized with `center == 1`.
"""
function TDVPEngine(psi::MPS{T}, H::MPO{T}) where {T}
    center(psi) == 1 ||
        error("psi must have center == 1; got $(center(psi)). " *
              "Call orthogonalize!(psi) first.")
    op_env = OperatorEnv(psi, psi, H; init_center=1)
    return TDVPEngine{T}(psi, H, op_env)
end

"""
    sweep!(engine::TDVPEngine, dt; max_dim=nothing, cutoff=0.0, num_center=1,
                                   krylovdim=30, tol=1e-12) -> avg_trunc

Perform one full sweep (right then left), applying `exp(-dt * H)` to psi.
Returns the average truncation error (0 for 1-site).
"""
function sweep!(engine::TDVPEngine{T}, dt::Number;
                max_dim::Union{Int,Nothing}=nothing,
                cutoff::Real=0.0,
                num_center::Int=1,
                krylovdim::Int=30,
                tol::Real=1e-12) where {T}
    num_center in (1, 2) || error("num_center must be 1 or 2.")
    center(engine.psi) == 1 ||
        error("psi.center must be 1 at the start of each sweep; " *
              "got $(center(engine.psi)).")
    N = length(engine.psi)
    truncs = Float64[]

    if num_center == 1
        for p in 1:N
            _update_1site!(engine, p, dt, "right", krylovdim, tol)
        end
        for p in N:-1:1
            _update_1site!(engine, p, dt, "left", krylovdim, tol)
        end
    else
        for p in 1:N-1
            tr = _update_2site!(engine, p, dt, max_dim, cutoff, "right",
                                krylovdim, tol)
            push!(truncs, tr)
        end
        for p in (N-1):-1:1
            tr = _update_2site!(engine, p, dt, max_dim, cutoff, "left",
                                krylovdim, tol)
            push!(truncs, tr)
        end
    end

    return isempty(truncs) ? 0.0 : sum(truncs) / length(truncs)
end

# ---------------------------------------------------------------------------
# Local helpers
# ---------------------------------------------------------------------------

# exp(t * H_eff) * phi via KrylovKit
function _expm_apply(apply_phi::Function, t::Number, phi::AbstractArray;
                     krylovdim::Int, tol::Real)
    sz = size(phi)
    f  = v -> vec(apply_phi(reshape(v, sz)))
    kd = min(krylovdim, length(phi))
    result, info = exponentiate(f, t, vec(phi);
                                krylovdim=kd, tol=tol, ishermitian=true)
    return reshape(result, sz)
end

# ---------------------------------------------------------------------------
# 1-site TDVP local update
# ---------------------------------------------------------------------------

function _update_1site!(engine::TDVPEngine{T}, p::Int, dt::Number,
                        absorb::String, krylovdim::Int, tol::Real) where {T}
    psi = engine.psi
    N   = length(psi)

    # 1. Prepare envs
    update_envs!(engine.op_env, p, p)
    Lenv = getenv(engine.op_env, p - 1)
    Renv = getenv(engine.op_env, p + 1)

    # 2. Forward propagation: phi = exp(-dt/2 * H_eff_1site) * psi[p]
    effH = EffOperator(Lenv, Renv, engine.H[p])
    phi  = make_phi(psi, p; n=1)
    phi  = _expm_apply(v -> apply(effH, v), -0.5 * dt, phi;
                       krylovdim=krylovdim, tol=tol)

    # Boundary: no backward step
    is_boundary = (absorb == "right" && p == N) || (absorb == "left" && p == 1)
    if is_boundary
        # phi is rank 3, shape (l, i, r). Just store it back.
        psi[p] = phi
        psi.center_left  = p
        psi.center_right = p
        return
    end

    # 3. SVD split phi into A (isometry) and C (bond tensor)
    if absorb == "right"
        # row=(l, i): n_row_legs=2 ; absorb singular values into Vt → C
        A, C, _ = svd_split(phi, 2; absorb="right")
        # A shape: (l, i, χ) ; C shape: (χ, r)
    else
        # row=(l,) : n_row_legs=1 ; absorb singular values into U → C
        C, A, _ = svd_split(phi, 1; absorb="left")
        # C shape: (l, χ) ; A shape: (χ, i, r)
    end

    # 4. Build 0-site env from op_env and A (do NOT update op_env)
    if absorb == "right"
        # New left env at bond between p and p+1
        new_L = grow_left_op(Lenv, A, engine.H[p], A)
        effH_0 = EffOperator(new_L, Renv)
    else
        new_R = grow_right_op(Renv, A, engine.H[p], A)
        effH_0 = EffOperator(Lenv, new_R)
    end

    # 5. Backward propagation: C' = exp(+dt/2 * H_eff_0site) * C
    C = _expm_apply(v -> apply(effH_0, v), +0.5 * dt, C;
                    krylovdim=krylovdim, tol=tol)

    # 6. Absorb C into neighbour and update psi
    if absorb == "right"
        # C shape (χ, r). Contract with psi[p+1] (l=r of A → χ).
        Anext = psi[p+1]
        @tensor new_nb[x, i, r] := C[x, y] * Anext[y, i, r]
        set_sites!(psi, Dict(p => A, p+1 => new_nb))
        psi.center_left  = p + 1
        psi.center_right = p + 1
    else
        # C shape (l, χ). Contract with psi[p-1].
        Aprev = psi[p-1]
        @tensor new_nb[l, i, x] := Aprev[l, i, y] * C[y, x]
        set_sites!(psi, Dict(p-1 => new_nb, p => A))
        psi.center_left  = p - 1
        psi.center_right = p - 1
    end
end

# ---------------------------------------------------------------------------
# 2-site TDVP local update
# ---------------------------------------------------------------------------

function _update_2site!(engine::TDVPEngine{T}, p::Int, dt::Number,
                        max_dim, cutoff, absorb::String,
                        krylovdim::Int, tol::Real) where {T}
    psi = engine.psi
    N   = length(psi)

    # 1. Prepare envs
    update_envs!(engine.op_env, p, p + 1)
    Lenv = getenv(engine.op_env, p - 1)
    Renv = getenv(engine.op_env, p + 2)

    # 2. Forward 2-site
    effH2 = EffOperator(Lenv, Renv, engine.H[p], engine.H[p+1])
    phi2  = make_phi(psi, p; n=2)
    phi2  = _expm_apply(v -> apply(effH2, v), -0.5 * dt, phi2;
                        krylovdim=krylovdim, tol=tol)

    # 3. SVD split + write back via update_sites!
    trunc = update_sites!(psi, p, phi2;
                          max_dim=max_dim, cutoff=cutoff, absorb=absorb)

    # Boundary: no backward
    is_boundary = (absorb == "right" && p == N - 1) || (absorb == "left" && p == 1)
    if is_boundary
        return Float64(trunc)
    end

    # 4. Backward 1-site evolve on the new center tensor
    if absorb == "right"
        q_back = p + 1
        # Recompute LR[p] (left env) using the freshly updated psi[p].
        update_envs!(engine.op_env, p + 1, p + 1)
        Lq = getenv(engine.op_env, p)
        Rq = getenv(engine.op_env, p + 2)
        effH1 = EffOperator(Lq, Rq, engine.H[q_back])
    else
        q_back = p
        update_envs!(engine.op_env, p, p)
        Lq = getenv(engine.op_env, p - 1)
        Rq = getenv(engine.op_env, p + 1)
        effH1 = EffOperator(Lq, Rq, engine.H[q_back])
    end

    phi_back = make_phi(psi, q_back; n=1)
    phi_back = _expm_apply(v -> apply(effH1, v), +0.5 * dt, phi_back;
                           krylovdim=krylovdim, tol=tol)
    psi[q_back] = phi_back
    psi.center_left  = q_back
    psi.center_right = q_back

    return Float64(trunc)
end
