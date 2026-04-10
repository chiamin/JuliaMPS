# MPOCompression.jl
#
# SVD-based compression for MPO.

"""
    _svd_two_mpo_sites(left, right; absorb, maxdim, cutoff)
        -> (left_new, right_new, discarded)

Merge two adjacent MPO site tensors, SVD-truncate, and split back.
Both `left` and `right` have shape (l, ip, i, r).
Returns left_new shape (l, ip1, i1, χ) and right_new shape (χ, ip2, i2, r).
"""
function _svd_two_mpo_sites(left::Array{T,4}, right::Array{T,4};
                            absorb::String,
                            maxdim::Int=typemax(Int),
                            cutoff::Real=0.0) where {T<:Number}
    @tensor aa[l, ip1, i1, ip2, i2, r] :=
        left[l, ip1, i1, x] * right[x, ip2, i2, r]
    # SVD with row=(l, ip1, i1): n_row_legs = 3
    A0, A1, discarded = svd_split(aa, 3; maxdim=maxdim, cutoff=cutoff, absorb=absorb)
    # A0 shape: (l, ip1, i1, χ);  A1 shape: (χ, ip2, i2, r)
    return A0, A1, discarded
end

"""
    svd_compress_mpo(W; max_dim=nothing, cutoff=0.0) -> MPO

Compress an MPO via SVD truncation (two-pass sweep).  First left-to-right
without truncation (left-canonicalize), then right-to-left with truncation.
"""
function svd_compress_mpo(W::MPO{T};
                          max_dim::Union{Int,Nothing}=nothing,
                          cutoff::Real=0.0) where {T}
    H = copy(W)
    N = length(H)
    N <= 1 && return H
    md = max_dim === nothing ? typemax(Int) : max_dim

    # Left-to-right: no truncation
    for p in 1:(N-1)
        left_new, right_new, _ = _svd_two_mpo_sites(
            H[p], H[p+1];
            absorb="right", maxdim=typemax(Int), cutoff=0.0,
        )
        H[p]   = left_new
        H[p+1] = right_new
    end

    # Right-to-left: truncate
    for p in (N-1):-1:1
        left_new, right_new, _ = _svd_two_mpo_sites(
            H[p], H[p+1];
            absorb="left", maxdim=md, cutoff=cutoff,
        )
        H[p]   = left_new
        H[p+1] = right_new
    end

    return H
end
