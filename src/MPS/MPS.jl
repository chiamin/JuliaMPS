# MPS.jl
#
# Open-boundary Matrix Product State, dense-only.
#
# Tensor leg order (fixed):
#   site tensor : Array{T,3}, dims = (l, i, r)
#     l = left virtual bond,  i = physical,  r = right virtual bond
#
# Open-chain boundary: size(_tensors[1], 1) == 1 and size(_tensors[end], 3) == 1.
#
# Canonical window [center_left, center_right] (1-indexed):
#   sites < center_left  are left-orthonormal
#   sites > center_right are right-orthonormal
#
# Callbacks: registered functions are called as `f(site::Int)` after every
# successful site update.  Used by DMRG environments to invalidate stale envs.
# Unlike the Python version, Julia callbacks are plain functions (no weakrefs).

mutable struct MPS{T<:Number}
    # WARNING: Do NOT write to `_tensors` directly (e.g. psi._tensors[i] = A).
    # Always use `psi[i] = A` (setindex!) or `set_sites!` so that callbacks
    # fire and any registered environments are properly invalidated.
    _tensors     :: Vector{Array{T,3}}
    center_left  :: Int
    center_right :: Int
    callbacks    :: Vector{Any}   # callable objects, called as f(site)
end

# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

"""
    MPS(tensors::Vector{Array{T,3}}) -> MPS{T}

Construct an MPS from a list of rank-3 site tensors with leg order (l, i, r).
Validates ranks, boundary bonds, and neighbour bond consistency.
"""
function MPS(tensors::Vector{Array{T,3}}) where {T<:Number}
    isempty(tensors) && error("An MPS must contain at least one site tensor.")
    psi = MPS{T}(copy(tensors), 1, length(tensors), Any[])
    _validate_bonds(psi)
    return psi
end

# ---------------------------------------------------------------------------
# Basic interface
# ---------------------------------------------------------------------------

Base.length(psi::MPS) = length(psi._tensors)
Base.lastindex(psi::MPS) = length(psi._tensors)
Base.firstindex(psi::MPS) = 1
Base.getindex(psi::MPS, site::Int) = psi._tensors[site]
Base.eltype(::MPS{T}) where {T} = T

function Base.setindex!(psi::MPS{T}, A::Array{T,3}, site::Int) where {T}
    old_tensor = psi._tensors[site]
    old_left   = psi.center_left
    old_right  = psi.center_right
    psi._tensors[site] = A
    psi.center_left  = min(psi.center_left, site)
    psi.center_right = max(psi.center_right, site)
    try
        check_mps_tensor(psi, site)
    catch e
        psi._tensors[site] = old_tensor
        psi.center_left  = old_left
        psi.center_right = old_right
        rethrow(e)
    end
    _notify(psi, site)
    return A
end

"""
    set_sites!(psi, updates::Dict{Int, Array{T,3}})

Atomically replace multiple site tensors.  All tensors are written first,
then validated, then all observers are notified once.  This avoids the
intermediate bond mismatch from updating adjacent sites one at a time.
"""
function set_sites!(psi::MPS{T}, updates::Dict{Int, <:Array{T,3}}) where {T}
    sites = sort(collect(keys(updates)))
    for s in sites
        psi._tensors[s] = updates[s]
    end
    for s in sites
        psi.center_left  = min(psi.center_left, s)
        psi.center_right = max(psi.center_right, s)
    end
    for s in sites
        check_mps_tensor(psi, s)
    end
    for s in sites
        _notify(psi, s)
    end
    return psi
end

"""
    register_callback!(psi, f)

Register `f` to be called as `f(site::Int)` whenever a site tensor is updated.
"""
function register_callback!(psi::MPS, f)
    push!(psi.callbacks, f)
    return psi
end

function _notify(psi::MPS, site::Int)
    for cb in psi.callbacks
        cb(site)
    end
end

# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------

phys_dims(psi::MPS) = [size(A, 2) for A in psi._tensors]
bond_dims(psi::MPS) = [size(psi._tensors[1], 1); [size(A, 3) for A in psi._tensors]]
max_dim(psi::MPS) = maximum(bond_dims(psi))

"""
    center(psi) -> Int or nothing

Returns the single-site center index when center_left == center_right,
or `nothing` when the canonical window spans multiple sites.
"""
function center(psi::MPS)
    return psi.center_left == psi.center_right ? psi.center_left : nothing
end

function Base.copy(psi::MPS{T}) where {T}
    new_psi = MPS{T}([copy(A) for A in psi._tensors],
                     psi.center_left, psi.center_right, Any[])
    return new_psi
end

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

"""
    check_mps_tensor(psi, site, tensor=nothing)

Validate that `tensor` (or `psi[site]` if nothing) is a valid MPS site tensor:
rank-3, neighbour bond dimensions match.
"""
function check_mps_tensor(psi::MPS, site::Int, tensor::Union{Array,Nothing}=nothing)
    A = tensor === nothing ? psi._tensors[site] : tensor
    ndims(A) == 3 || error("Site $site must be rank-3; got rank $(ndims(A)).")
    if site > 1
        size(A, 1) == size(psi._tensors[site-1], 3) ||
            error("Bond mismatch between sites $(site-1) and $site: " *
                  "$(size(psi._tensors[site-1], 3)) vs $(size(A, 1)).")
    end
    if site < length(psi)
        size(A, 3) == size(psi._tensors[site+1], 1) ||
            error("Bond mismatch between sites $site and $(site+1): " *
                  "$(size(A, 3)) vs $(size(psi._tensors[site+1], 1)).")
    end
    return nothing
end

function _validate_bonds(psi::MPS)
    for site in 1:length(psi)
        check_mps_tensor(psi, site)
    end
    size(psi._tensors[1], 1) == 1 || error("First tensor left bond must be 1.")
    size(psi._tensors[end], 3) == 1 || error("Last tensor right bond must be 1.")
    (1 <= psi.center_left <= psi.center_right <= length(psi)) ||
        error("Invalid center window [$(psi.center_left), $(psi.center_right)].")
    return nothing
end

"""
    check_left_right_orthonormal(psi; atol=1e-10)

Verify that sites < center_left are left-orthonormal and sites > center_right
are right-orthonormal.  Raises an error on the first violation.
"""
function check_left_right_orthonormal(psi::MPS{T}; atol::Real=1e-10) where {T}
    for site in 1:length(psi)
        A = psi._tensors[site]
        ld, pd, rd = size(A)
        if site < psi.center_left
            M = reshape(A, ld * pd, rd)
            G = M' * M
            isapprox(G, Matrix{T}(I, rd, rd); atol=atol) ||
                error("Site $site is not left-orthonormal " *
                      "(required for sites < center_left=$(psi.center_left)).")
        end
        if site > psi.center_right
            M = reshape(A, ld, pd * rd)
            G = M * M'
            isapprox(G, Matrix{T}(I, ld, ld); atol=atol) ||
                error("Site $site is not right-orthonormal " *
                      "(required for sites > center_right=$(psi.center_right)).")
        end
    end
    return nothing
end

# ---------------------------------------------------------------------------
# Norm / normalization
# ---------------------------------------------------------------------------

"""
    norm(psi::MPS) -> Real

Compute sqrt(real(<psi|psi>)) by left-to-right contraction.
"""
function LinearAlgebra.norm(psi::MPS{T}) where {T}
    env = ones(T, 1, 1)   # (dn=χ_bra, up=χ_ket) — both side share psi
    for k in 1:length(psi)
        A = psi._tensors[k]
        @tensor env_new[dn, up] := env[x, y] * A[x, i, dn] * conj(A)[y, i, up]
        env = env_new
    end
    val = only(env)
    return sqrt(real(val))
end

"""
    normalize!(psi::MPS) -> psi

Scale the orthogonality center so that ||psi|| = 1.
Requires a single orthogonality center; call `orthogonalize!` first if needed.
"""
function LinearAlgebra.normalize!(psi::MPS)
    c = center(psi)
    c === nothing && error("MPS has no single orthogonality center; " *
                            "call orthogonalize! first.")
    nrm = norm(psi)
    nrm < 1e-14 && error("Cannot normalize a zero MPS.")
    psi[c] = psi._tensors[c] .* (1 / nrm)
    return psi
end

# ---------------------------------------------------------------------------
# Orthogonality center movement
# ---------------------------------------------------------------------------

"""
    _shift_center_right!(psi, site)

SVD with row=(l,i), absorb singular values into Vt, push Vt into site+1.
After this call, site is left-orthonormal and the center moves to site+1.
"""
function _shift_center_right!(psi::MPS{T}, site::Int) where {T}
    A = psi._tensors[site]
    U, Vt, _ = svd_split(A, 2; absorb="right")
    # U shape: (l, i, χ) → new psi[site]
    # Vt shape: (χ, r)   → contract into psi[site+1]
    Anext = psi._tensors[site+1]
    @tensor right_new[x, i, r] := Vt[x, y] * Anext[y, i, r]
    set_sites!(psi, Dict(site => U, site+1 => right_new))
    psi.center_left  = site + 1
    psi.center_right = site + 1
    return psi
end

"""
    _shift_center_left!(psi, site)

SVD with row=(l,), absorb singular values into U, push U into site-1.
After this call, site is right-orthonormal and the center moves to site-1.
"""
function _shift_center_left!(psi::MPS{T}, site::Int) where {T}
    A = psi._tensors[site]
    US, Vt, _ = svd_split(A, 1; absorb="left")
    # US shape: (l, χ)    → contract into psi[site-1]
    # Vt shape: (χ, i, r) → new psi[site]
    Aprev = psi._tensors[site-1]
    @tensor left_new[l, i, x] := Aprev[l, i, y] * US[y, x]
    set_sites!(psi, Dict(site-1 => left_new, site => Vt))
    psi.center_left  = site - 1
    psi.center_right = site - 1
    return psi
end

"""
    orthogonalize!(psi; center=nothing) -> psi

Canonicalize the active center window.  After this routine the window
collapses to a single center at center_left.  If `center` is provided,
the orthogonality center is then moved there.
"""
function orthogonalize!(psi::MPS; center::Union{Int,Nothing}=nothing)
    if center !== nothing
        (1 <= center <= length(psi)) ||
            error("Center site $center out of range [1, $(length(psi))].")
    end
    for site in psi.center_right:-1:psi.center_left+1
        _shift_center_left!(psi, site)
    end
    _validate_bonds(psi)
    if center !== nothing && psi.center_left != center
        move_center!(psi, center)
    end
    return psi
end

"""
    move_center!(psi, site) -> psi

Move the orthogonality center to `site`.  If the MPS has no single center,
first canonicalize via `orthogonalize!`.
"""
function move_center!(psi::MPS, site::Int)
    (1 <= site <= length(psi)) ||
        error("Center site $site out of range [1, $(length(psi))].")
    if center(psi) === nothing
        orthogonalize!(psi)
    end
    while psi.center_left < site
        _shift_center_right!(psi, psi.center_left)
    end
    while psi.center_left > site
        _shift_center_left!(psi, psi.center_left)
    end
    return psi
end

# ---------------------------------------------------------------------------
# make_phi / update_sites
# ---------------------------------------------------------------------------

"""
    make_phi(psi, p; n=1) -> Array

Merge `n` consecutive site tensors starting at `p` into a single tensor.

- n=1: returns shape `(l, i0, r)` (a copy of psi[p])
- n=2: returns shape `(l, i0, i1, r)`
"""
function make_phi(psi::MPS{T}, p::Int; n::Int=1) where {T}
    (1 <= p <= length(psi)) ||
        error("Site $p out of range [1, $(length(psi))].")
    n >= 1 || error("n must be >= 1.")
    p + n - 1 <= length(psi) ||
        error("Sites $p..$(p+n-1) exceed MPS length $(length(psi)).")

    if n == 1
        return copy(psi._tensors[p])
    elseif n == 2
        A = psi._tensors[p]
        B = psi._tensors[p+1]
        @tensor phi[l, i0, i1, r] := A[l, i0, x] * B[x, i1, r]
        return phi
    else
        error("make_phi supports n=1 or n=2; got n=$n.")
    end
end

"""
    update_sites!(psi, p, phi; max_dim=nothing, cutoff=0.0, absorb="right") -> Real

Decompose `phi` back into site tensors and update psi in-place.
Inverse of `make_phi`.

- 1-site phi (rank 3, shape (l,i,r)): updates psi[p] (and absorbs into neighbour
  according to `absorb`).
- 2-site phi (rank 4, shape (l,i0,i1,r)): updates psi[p] and psi[p+1].

`absorb="right"` moves the center to the right; `absorb="left"` moves it left.
Returns the discarded weight from the SVD.
"""
function update_sites!(psi::MPS{T}, p::Int, phi::Array{T};
                       max_dim::Union{Int,Nothing}=nothing,
                       cutoff::Real=0.0,
                       absorb::String="right") where {T}
    absorb in ("left", "right") || error("absorb must be 'left' or 'right'.")
    md = max_dim === nothing ? typemax(Int) : max_dim
    n = ndims(phi) - 2   # subtract the (l, r) virtual legs

    if n == 1
        return _update_1site!(psi, p, phi, md, cutoff, absorb)
    elseif n == 2
        return _update_2site!(psi, p, phi, md, cutoff, absorb)
    else
        error("update_sites! supports n=1 or 2 (phi rank 3 or 4); got rank $(ndims(phi)).")
    end
end

function _update_1site!(psi::MPS{T}, p, phi, md, cutoff, absorb) where {T}
    if absorb == "right"
        # row=(l, i): n_row_legs=2
        U, Vt, discarded = svd_split(phi, 2; maxdim=md, cutoff=cutoff, absorb="right")
        # U:  (l, i, χ)  → new psi[p]
        # Vt: (χ, r)     → absorb into psi[p+1]
        Anext = psi._tensors[p+1]
        @tensor next_new[x, i, r] := Vt[x, y] * Anext[y, i, r]
        set_sites!(psi, Dict(p => U, p+1 => next_new))
        psi.center_left  = p + 1
        psi.center_right = p + 1
        return discarded
    else  # absorb == "left"
        # row=(l,): n_row_legs=1
        US, Vt, discarded = svd_split(phi, 1; maxdim=md, cutoff=cutoff, absorb="left")
        # US: (l, χ)     → absorb into psi[p-1]
        # Vt: (χ, i, r)  → new psi[p]
        Aprev = psi._tensors[p-1]
        @tensor prev_new[l, i, x] := Aprev[l, i, y] * US[y, x]
        set_sites!(psi, Dict(p-1 => prev_new, p => Vt))
        psi.center_left  = p - 1
        psi.center_right = p - 1
        return discarded
    end
end

function _update_2site!(psi::MPS{T}, p, phi, md, cutoff, absorb) where {T}
    # phi shape (l, i0, i1, r); SVD with row=(l, i0): n_row_legs=2
    A0, A1, discarded = svd_split(phi, 2; maxdim=md, cutoff=cutoff, absorb=absorb)
    # A0: (l, i0, χ) → new psi[p]
    # A1: (χ, i1, r) → new psi[p+1]
    set_sites!(psi, Dict(p => A0, p+1 => A1))
    cnew = absorb == "right" ? p+1 : p
    psi.center_left  = cnew
    psi.center_right = cnew
    return discarded
end
