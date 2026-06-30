# Engine.jl
#
# DMRG sweep engine using KrylovKit's eigsolve as the local eigensolver.
#
# Usage:
#     engine = DMRGEngine(psi, H)
#     for (D, cut) in zip([20, 50, 100], [0.0, 0.0, 1e-8])
#         E, trunc = sweep!(engine; max_dim=D, cutoff=cut, num_center=2)
#     end
#
# `psi` is modified in-place.  Environments are built once at construction
# and reused across sweeps.

mutable struct DMRGEngine{T<:Number}
    psi          :: MPS{T}
    H            :: MPO{T}
    op_env       :: OperatorEnv{T}
    ortho_states :: Vector{MPS{T}}
    ortho_weights:: Vector{Float64}
    vec_envs     :: Vector{VectorEnv{T}}
end

"""
    DMRGEngine(psi, H; ortho_states=MPS[], ortho_weights=Float64[]) -> DMRGEngine

Build the engine.  `psi` must have a single orthogonality center at site 1
(right-canonical form).
"""
function DMRGEngine(psi::MPS{T}, H::MPO{T};
                    ortho_states::Vector{<:MPS}=MPS{T}[],
                    ortho_weights::Vector{<:Real}=Float64[]) where {T}
    center(psi) == 1 ||
        error("psi must have center == 1; got center=$(center(psi)). " *
              "Call orthogonalize!(psi) first.")
    length(ortho_states) == length(ortho_weights) ||
        error("ortho_states and ortho_weights must have the same length.")

    op_env = OperatorEnv(psi, psi, H; init_center=1)
    vec_envs = [VectorEnv(s, psi; init_center=1) for s in ortho_states]
    return DMRGEngine{T}(psi, H, op_env,
                         collect(MPS{T}, ortho_states),
                         Float64.(ortho_weights), vec_envs)
end

"""
    sweep!(engine; max_dim=nothing, cutoff=0.0, num_center=2,
                   krylovdim=30, tol=1e-12) -> (E, avg_trunc)

Perform one full DMRG sweep (right then left).  Returns the energy after
the final local optimization and the average truncation error.
"""
function sweep!(engine::DMRGEngine{T};
                max_dim::Union{Int,Nothing}=nothing,
                cutoff::Real=0.0,
                num_center::Int=2,
                krylovdim::Int=30,
                tol::Real=1e-12) where {T}
    num_center in (1, 2) || error("num_center must be 1 or 2.")
    center(engine.psi) == 1 ||
        error("psi.center must be 1 at the start of each sweep; " *
              "got $(center(engine.psi)).")
    N = length(engine.psi)
    n = num_center
    energies = Float64[]
    truncs   = Float64[]

    # Sweep right: p = 1 .. N-1
    for p in 1:N-1
        E, tr = _local_update!(engine, p, n, max_dim, cutoff, "right",
                               krylovdim, tol)
        push!(energies, E); push!(truncs, tr)
    end

    # Sweep left
    # 2-site: p = N-1 .. 1
    # 1-site: p = N   .. 2
    left_start = N - n + 1
    left_stop  = n == 2 ? 1 : 2
    for p in left_start:-1:left_stop
        E, tr = _local_update!(engine, p, n, max_dim, cutoff, "left",
                               krylovdim, tol)
        push!(energies, E); push!(truncs, tr)
    end

    # Truncation at the SVD steps discards a little weight, so after the sweep
    # the state norm is slightly below 1.  Renormalize once here: the MPS has a
    # single orthogonality center at the end of the sweep, so normalize! simply
    # rescales that center tensor.  (Energy is a Rayleigh quotient and is
    # unaffected; this only fixes the overall scale ⟨ψ|ψ⟩ = 1.)
    LinearAlgebra.normalize!(engine.psi)

    avg_trunc = isempty(truncs) ? 0.0 : sum(truncs) / length(truncs)
    return energies[end], avg_trunc
end

function _local_update!(engine::DMRGEngine{T}, p::Int, n::Int,
                        max_dim, cutoff, absorb::String,
                        krylovdim::Int, tol::Real) where {T}
    psi = engine.psi
    update_envs!(engine.op_env, p, p + n - 1)
    for ve in engine.vec_envs
        update_envs!(ve, p, p + n - 1)
    end

    Lenv = getenv(engine.op_env, p - 1)
    Renv = getenv(engine.op_env, p + n)
    mpo_tensors = ntuple(k -> engine.H[p + k - 1], n)
    effH = EffOperator(Lenv, Renv, mpo_tensors...)

    for (ve, ortho, w) in zip(engine.vec_envs, engine.ortho_states, engine.ortho_weights)
        Lvec = getenv(ve, p - 1)
        Rvec = getenv(ve, p + n)
        mps_tensors = ntuple(k -> ortho[p + k - 1], n)
        eff_vec = EffVector(Lvec, Rvec, mps_tensors...)
        add_term!(effH, eff_vec, w)
    end

    phi = make_phi(psi, p; n=n)
    sz = size(phi)
    f = v -> vec(apply(effH, reshape(v, sz)))
    kd = min(krylovdim, length(phi))
    vals, vecs, info = eigsolve(f, vec(phi), 1, :SR;
                                krylovdim=kd, tol=tol, ishermitian=true)
    E = real(vals[1])
    phi_new = reshape(vecs[1], sz)
    trunc = update_sites!(psi, p, phi_new;
                          max_dim=max_dim, cutoff=cutoff, absorb=absorb)
    return E, Float64(trunc)
end

"""
    dmrg!(psi, H; sweeps, max_dims, cutoffs, num_center=2, ...) -> Vector{Float64}

Convenience wrapper: build a DMRGEngine and run multiple sweeps with
per-sweep `max_dim` and `cutoff` schedules.  Returns the energy after each
sweep.
"""
function dmrg!(psi::MPS, H::MPO;
               sweeps::Int,
               max_dims=nothing,
               cutoffs=nothing,
               num_center::Int=2,
               kwargs...)
    md = max_dims === nothing ? fill(typemax(Int), sweeps) : collect(max_dims)
    ct = cutoffs  === nothing ? fill(0.0, sweeps) : collect(cutoffs)
    length(md) == sweeps || error("max_dims must have length sweeps.")
    length(ct) == sweeps || error("cutoffs must have length sweeps.")
    engine = DMRGEngine(psi, H; kwargs...)
    energies = Float64[]
    for s in 1:sweeps
        E, _ = sweep!(engine; max_dim=md[s], cutoff=ct[s], num_center=num_center)
        push!(energies, E)
    end
    return energies
end
