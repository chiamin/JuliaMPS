# Compression.jl
#
# SVD-based compression for MPS.

"""
    _svd_two_sites(left, right; absorb, maxdim, cutoff)
        -> (left_new, right_new, discarded)

Merge two adjacent MPS site tensors, SVD-truncate, and split back.
Both `left` and `right` have shape (l, i, r).
"""
function _svd_two_sites(left::Array{T,3}, right::Array{T,3};
                        absorb::String,
                        maxdim::Int=typemax(Int),
                        cutoff::Real=0.0) where {T<:Number}
    @tensor aa[l, i1, i2, r] := left[l, i1, x] * right[x, i2, r]
    A0, A1, discarded = svd_split(aa, 2; maxdim=maxdim, cutoff=cutoff, absorb=absorb)
    # A0 shape: (l, i1, χ),  A1 shape: (χ, i2, r)
    return A0, A1, discarded
end

"""
    svd_compress_mps(psi; max_dim=nothing, cutoff=0.0) -> MPS

Compress an MPS via SVD truncation.  Moves the orthogonality center to the
last site (so that singular values on each bond equal Schmidt values), then
performs a single right-to-left SVD sweep with truncation.

`psi` is not modified; a new MPS is returned.
"""
function svd_compress_mps(psi::MPS{T};
                          max_dim::Union{Int,Nothing}=nothing,
                          cutoff::Real=0.0) where {T}
    phi = copy(psi)
    move_center!(phi, length(phi))
    N = length(phi)
    md = max_dim === nothing ? typemax(Int) : max_dim

    for p in (N-1):-1:1
        left_new, right_new, _ = _svd_two_sites(
            phi[p], phi[p+1];
            absorb="left", maxdim=md, cutoff=cutoff,
        )
        set_sites!(phi, Dict(p => left_new, p+1 => right_new))
        phi.center_left  = p
        phi.center_right = p
    end
    return phi
end
