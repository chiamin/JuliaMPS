# EffectiveOperators.jl
#
# Effective operators for DMRG local subspace optimisation.
#
# EffOperator
#   Projects an MPO into the effective subspace at site(s) p..p+n-1.
#   Supports n=0 (bond tensor for TDVP), n=1 (1-site DMRG), n=2 (2-site DMRG).
#   `apply(effH, phi)` computes H_eff |phi⟩.
#
# EffVector
#   Projects a reference MPS into the effective subspace.
#   `inner(eff_vec, phi)` returns <Φ_0|phi⟩.
#
# Tensor leg-order conventions:
#   L, R (op env)  : (dn = ket virtual, w = MPO virtual, up = bra virtual)
#   L, R (vec env) : (dn = ket virtual, up = bra virtual)
#   phi (1-site)   : (l, i, r)             — same shape as MPS site tensor
#   phi (2-site)   : (l, i0, i1, r)
#   phi (0-site)   : (l, r)                — bond tensor

# ---------------------------------------------------------------------------
# EffOperator
# ---------------------------------------------------------------------------

mutable struct EffOperator{T<:Number}
    L           :: Array{T,3}    # (dn, w, up)
    R           :: Array{T,3}
    mpo_tensors :: Vector{Array{T,4}}
    # Penalty terms: list of (eff_vec, weight)
    terms       :: Vector{Tuple{Any, T}}
end

"""
    EffOperator(L, R, mpo_tensors...) -> EffOperator

Build the effective operator from environments and MPO tensors.

- 0 mpo tensors → 0-site (bond tensor; phi has rank 2)
- 1 mpo tensor  → 1-site DMRG
- 2 mpo tensors → 2-site DMRG
"""
function EffOperator(L::Array{T,3}, R::Array{T,3},
                     mpo_tensors::Array{T,4}...) where {T<:Number}
    return EffOperator{T}(L, R, collect(mpo_tensors),
                          Tuple{Any, T}[])
end

"""
    add_term!(effH, eff_vec, weight)

Add a weighted rank-1 term `weight * |Φ_0⟩⟨Φ_0|` to H_eff.
Used for excited-state targeting.
"""
function add_term!(effH::EffOperator{T}, eff_vec, weight::Number) where {T}
    push!(effH.terms, (eff_vec, T(weight)))
    return effH
end

"""
    apply(effH, phi) -> Array

Compute `H_eff |phi⟩` (including any added rank-1 penalty terms).
The result has the same shape as `phi`.
"""
function apply(effH::EffOperator{T}, phi::Array{T}) where {T}
    out = _apply_operator(effH, phi)
    for (eff_vec, weight) in effH.terms
        ov = inner(eff_vec, phi)
        out .+= (weight * ov) .* eff_vec.tensor
    end
    return out
end

# 1-, 2-, 0-site contractions
function _apply_operator(effH::EffOperator{T}, phi::Array{T,3}) where {T}
    length(effH.mpo_tensors) == 1 ||
        error("phi rank 3 expects 1 MPO tensor; got $(length(effH.mpo_tensors)).")
    L = effH.L; R = effH.R
    M = effH.mpo_tensors[1]
    @tensor out[lo, ip, ro] :=
        L[ldn, wL, lo] *
        phi[ldn, i, rdn] *
        M[wL, ip, i, wR] *
        R[rdn, wR, ro]
    return out
end

function _apply_operator(effH::EffOperator{T}, phi::Array{T,4}) where {T}
    length(effH.mpo_tensors) == 2 ||
        error("phi rank 4 expects 2 MPO tensors; got $(length(effH.mpo_tensors)).")
    L = effH.L; R = effH.R
    M0 = effH.mpo_tensors[1]
    M1 = effH.mpo_tensors[2]
    @tensor out[lo, ip0, ip1, ro] :=
        L[ldn, wL, lo] *
        phi[ldn, i0, i1, rdn] *
        M0[wL, ip0, i0, wM] *
        M1[wM, ip1, i1, wR] *
        R[rdn, wR, ro]
    return out
end

function _apply_operator(effH::EffOperator{T}, phi::Array{T,2}) where {T}
    length(effH.mpo_tensors) == 0 ||
        error("phi rank 2 expects 0 MPO tensors (bond tensor); got $(length(effH.mpo_tensors)).")
    L = effH.L; R = effH.R
    @tensor out[lo, ro] :=
        L[ldn, w, lo] * phi[ldn, rdn] * R[rdn, w, ro]
    return out
end

# ---------------------------------------------------------------------------
# EffVector
# ---------------------------------------------------------------------------

mutable struct EffVector{T<:Number, A<:AbstractArray}
    tensor :: A   # |Φ_0⟩ in ket form, same shape as phi
end

"""
    EffVector(L, R, mps_tensors...) -> EffVector

Pre-compute the projected reference vector |Φ_0⟩ by contracting the overlap
environments with the reference site tensors.
"""
function EffVector(L::Array{T,2}, R::Array{T,2},
                   mps_tensors::Array{T,3}...) where {T<:Number}
    n = length(mps_tensors)
    if n == 0
        # 0-site: contract L["dn"] with R["dn"]; output (l, r) from (up, up).
        @tensor out[lo, ro] := L[bond, lo] * R[bond, ro]
        return EffVector{T, typeof(out)}(out)
    elseif n == 1
        A = mps_tensors[1]
        @tensor out[lo, i, ro] := L[ldn, lo] * A[ldn, i, rdn] * R[rdn, ro]
        return EffVector{T, typeof(out)}(out)
    elseif n == 2
        A0 = mps_tensors[1]; A1 = mps_tensors[2]
        @tensor out[lo, i0, i1, ro] :=
            L[ldn, lo] * A0[ldn, i0, mid] * A1[mid, i1, rdn] * R[rdn, ro]
        return EffVector{T, typeof(out)}(out)
    else
        error("EffVector supports n=0,1,2 site tensors; got $n.")
    end
end

"""
    inner(eff_vec, phi) -> Number

Compute `<Φ_0|phi⟩`.
"""
function inner(eff_vec::EffVector, phi::AbstractArray)
    return sum(conj.(eff_vec.tensor) .* phi)
end
