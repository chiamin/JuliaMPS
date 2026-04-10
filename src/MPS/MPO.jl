# MPO.jl
#
# Open-boundary Matrix Product Operator, dense-only.
#
# Tensor leg order (fixed):
#   site tensor : Array{T,4}, dims = (l, ip, i, r)
#     l  = left MPO virtual bond
#     ip = outgoing physical (bra side; contracted with conj(MPS))
#     i  = incoming physical (ket side; contracted with MPS)
#     r  = right MPO virtual bond
#
# Open-chain boundary: size(W[1], 1) == 1 and size(W[end], 4) == 1.
# Physical consistency: size(W[k], 2) == size(W[k], 3) at every site.

mutable struct MPO{T<:Number}
    # WARNING: Do NOT write to `_tensors` directly (e.g. W._tensors[i] = A).
    # Always use `W[i] = A` (setindex!) so that callbacks fire and any
    # registered environments are properly invalidated.
    _tensors  :: Vector{Array{T,4}}
    callbacks :: Vector{Any}
end

# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

"""
    MPO(tensors::Vector{Array{T,4}}) -> MPO{T}

Construct an MPO from a list of rank-4 site tensors with leg order (l, ip, i, r).
Validates ranks, physical bond consistency, neighbour virtual bonds, and boundary.
"""
function MPO(tensors::Vector{Array{T,4}}) where {T<:Number}
    isempty(tensors) && error("An MPO must contain at least one site tensor.")
    W = MPO{T}(copy(tensors), Any[])
    _validate_mpo_bonds(W)
    return W
end

# ---------------------------------------------------------------------------
# Basic interface
# ---------------------------------------------------------------------------

Base.length(W::MPO) = length(W._tensors)
Base.lastindex(W::MPO) = length(W._tensors)
Base.firstindex(W::MPO) = 1
Base.getindex(W::MPO, site::Int) = W._tensors[site]
Base.eltype(::MPO{T}) where {T} = T

function Base.setindex!(W::MPO{T}, A::Array{T,4}, site::Int) where {T}
    ndims(A) == 4 ||
        error("Site $site MPO tensor must be rank-4; got rank $(ndims(A)).")
    size(A, 2) == size(A, 3) ||
        error("Site $site: physical legs ip and i must match; got $(size(A,2)) vs $(size(A,3)).")
    W._tensors[site] = A
    for cb in W.callbacks
        cb(site)
    end
    return A
end

function register_callback!(W::MPO, f)
    push!(W.callbacks, f)
    return W
end

# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------

phys_dims(W::MPO) = [size(A, 3) for A in W._tensors]
mpo_dims(W::MPO) = [size(W._tensors[1], 1); [size(A, 4) for A in W._tensors]]

function Base.copy(W::MPO{T}) where {T}
    return MPO{T}([copy(A) for A in W._tensors], Any[])
end

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

function _validate_mpo_bonds(W::MPO)
    for (k, A) in enumerate(W._tensors)
        ndims(A) == 4 ||
            error("Site $k MPO tensor must be rank-4; got rank $(ndims(A)).")
        size(A, 2) == size(A, 3) ||
            error("Site $k: physical legs ip and i must match; " *
                  "got $(size(A,2)) vs $(size(A,3)).")
    end
    for k in 1:length(W)-1
        size(W._tensors[k], 4) == size(W._tensors[k+1], 1) ||
            error("Virtual bond mismatch between MPO sites $k and $(k+1): " *
                  "$(size(W._tensors[k], 4)) vs $(size(W._tensors[k+1], 1)).")
    end
    size(W._tensors[1], 1) == 1 ||
        error("W[1] left bond must have dim=1; got $(size(W._tensors[1], 1)).")
    size(W._tensors[end], 4) == 1 ||
        error("W[end] right bond must have dim=1; got $(size(W._tensors[end], 4)).")
    return nothing
end
