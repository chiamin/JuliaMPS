# TensorUtils.jl
#
# SVD/QR decompositions and direct sum for dense Arrays.
# No labels — leg order is the caller's responsibility.
#
# Conventions used throughout this project:
#   MPS tensor : Array{T,3}  dims = (l, i, r)
#   MPO tensor : Array{T,4}  dims = (l, ip, i, r)
#   Env tensor : Array{T,3}  dims = (dn, w, up)

# ---------------------------------------------------------------------------
# svd_split
# ---------------------------------------------------------------------------

"""
    svd_split(A, n_row_legs; maxdim, cutoff, absorb) -> (U, Vt, discarded)

SVD-decompose `A` by treating its first `n_row_legs` dimensions as the row
index and the remaining dimensions as the column index.

# Arguments
- `A`: input Array of any rank and element type.
- `n_row_legs`: number of leading dimensions to treat as the row index.
- `maxdim`: maximum number of singular values to keep (default: no limit).
- `cutoff`: discard singular values whose normalised rho eigenvalue
  `λ_i = |s_i|² / Σ|s_j|²` is below this threshold (default: 0).
  At least one singular value is always kept.
- `absorb`: where to absorb the singular values.
  `"right"` (default): multiply s into Vt.
  `"left"`: multiply s into U.
  `nothing`: return s separately as a diagonal vector; in this case the
  function returns `(U, s, Vt, discarded)`.

# Returns (when absorb ∈ {"left","right"})
- `U`  shape `(size(A)[1:n_row_legs]..., χ)`
- `Vt` shape `(χ, size(A)[n_row_legs+1:end]...)`
- `discarded`: normalised truncation error `max(0, 1 - Σ_kept λ_i)`

# Returns (when absorb === nothing)
- `U`  shape `(size(A)[1:n_row_legs]..., χ)`
- `s`  shape `(χ,)`
- `Vt` shape `(χ, size(A)[n_row_legs+1:end]...)`
- `discarded`
"""
function svd_split(A::AbstractArray{T}, n_row_legs::Int;
                   maxdim::Int = typemax(Int),
                   cutoff::Real = 0.0,
                   absorb::Union{String,Nothing} = "right") where T

    @assert 1 <= n_row_legs < ndims(A) "n_row_legs must be in [1, ndims(A)-1]"
    @assert absorb ∈ ("left", "right", nothing) "absorb must be \"left\", \"right\", or nothing"
    @assert cutoff >= 0 "cutoff must be >= 0"

    row_dims = size(A)[1:n_row_legs]
    col_dims = size(A)[n_row_legs+1:end]
    row_dim  = prod(row_dims)
    col_dim  = prod(col_dims)

    M = reshape(A, row_dim, col_dim)
    F = svd(M)
    s = F.S  # singular values, descending order

    # Normalised rho eigenvalues
    total_sq = sum(abs2, s)
    if total_sq == 0
        error("All singular values are zero; cannot normalise rho eigenvalues.")
    end
    lambda = abs2.(s) ./ total_sq

    # Find truncation index: keep largest singular values satisfying cutoff and maxdim
    chi = min(length(s), maxdim)
    # Remove from the tail while lambda[chi] < cutoff (always keep at least 1)
    while chi > 1 && lambda[chi] < cutoff
        chi -= 1
    end

    discarded = max(0.0, 1.0 - sum(lambda[1:chi]))

    U_mat  = F.U[:, 1:chi]
    s_kept = s[1:chi]
    Vt_mat = F.Vt[1:chi, :]

    # Reshape back to tensor form
    U_shape  = (row_dims..., chi)
    Vt_shape = (chi, col_dims...)
    U  = reshape(U_mat,  U_shape)
    Vt = reshape(Vt_mat, Vt_shape)

    if absorb === nothing
        return U, s_kept, Vt, discarded
    elseif absorb == "right"
        # absorb s into Vt: Vt[k,...] *= s[k]
        Vt_abs = Vt .* reshape(s_kept, chi, ones(Int, length(col_dims))...)
        return U, Vt_abs, discarded
    else  # absorb == "left"
        # absorb s into U: U[...,k] *= s[k]
        U_abs = U .* reshape(s_kept, ones(Int, length(row_dims))..., chi)
        return U_abs, Vt, discarded
    end
end

# ---------------------------------------------------------------------------
# qr_split
# ---------------------------------------------------------------------------

"""
    qr_split(A, n_row_legs) -> (Q, R)

QR-decompose `A` by treating its first `n_row_legs` dimensions as the row
index and the remaining dimensions as the column index.

# Returns
- `Q` shape `(size(A)[1:n_row_legs]..., χ)`  — isometry (Q†Q = I)
- `R` shape `(χ, size(A)[n_row_legs+1:end]...)`
where `χ = min(row_dim, col_dim)`.
"""
function qr_split(A::AbstractArray{T}, n_row_legs::Int) where T
    @assert 1 <= n_row_legs < ndims(A) "n_row_legs must be in [1, ndims(A)-1]"

    row_dims = size(A)[1:n_row_legs]
    col_dims = size(A)[n_row_legs+1:end]
    row_dim  = prod(row_dims)
    col_dim  = prod(col_dims)

    M = reshape(A, row_dim, col_dim)
    F = qr(M)
    Q = Matrix(F.Q)
    R = Matrix(F.R)

    chi = size(Q, 2)
    Q_shape = (row_dims..., chi)
    R_shape = (chi, col_dims...)

    return reshape(Q, Q_shape), reshape(R, R_shape)
end

# ---------------------------------------------------------------------------
# direct_sum
# ---------------------------------------------------------------------------

"""
    direct_sum(A, B, sum_dims) -> C

Compute the direct sum of two arrays `A` and `B` along the dimensions listed
in `sum_dims`. Dimensions not in `sum_dims` must have equal size in `A` and `B`
and are kept unchanged in the output.

The result `C` has:
- `size(C, d) = size(A, d) + size(B, d)` for `d ∈ sum_dims`
- `size(C, d) = size(A, d)`              for `d ∉ sum_dims`

Off-diagonal blocks (where one array's contribution lives) are zero.

# Example (MPS interior site, dims = (l, i, r))
    direct_sum(A, B, [1, 3])
    # C shape: (l_A+l_B, i, r_A+r_B)
    # C[1:l_A, :, 1:r_A]          = A
    # C[l_A+1:end, :, r_A+1:end]  = B
    # everything else              = 0
"""
function direct_sum(A::AbstractArray{T,N}, B::AbstractArray{S,N},
                    sum_dims) where {T,S,N}
    ET = promote_type(T, S)
    out_size = ntuple(N) do d
        if d in sum_dims
            size(A, d) + size(B, d)
        else
            @assert size(A, d) == size(B, d) "Non-summed dim $d must match: $(size(A,d)) vs $(size(B,d))"
            size(A, d)
        end
    end

    C = zeros(ET, out_size)

    # A occupies the "first" block along sum_dims
    idx_A = ntuple(d -> d in sum_dims ? (1:size(A, d)) : (1:size(A, d)), N)
    C[idx_A...] = A

    # B occupies the "second" block along sum_dims
    idx_B = ntuple(N) do d
        if d in sum_dims
            (size(A, d)+1):(size(A, d)+size(B, d))
        else
            1:size(B, d)
        end
    end
    C[idx_B...] = B

    return C
end
