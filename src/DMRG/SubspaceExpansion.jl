# SubspaceExpansion.jl
#
# Strictly-single-site (3S) subspace expansion for 1-site DMRG
# (Hubig, McCulloch, Schollwöck, PRB 91, 155115 (2015)).
#
# One-site DMRG optimises a single site tensor at a time and therefore cannot, by
# itself, grow a bond dimension the way a two-site update can: the local problem
# is confined to the current bond's fixed size.  3S expansion fixes this by
# ENLARGING the bond before the local SVD, mixing in the directions that H wants
# to explore.  Concretely, on a rightward move at site `p` it forms the expansion
# term
#
#     P[l, i, (r, w)] = sum_{l',ii}  Lenv[l', w, l] * A[l', ii, r] * W[w, i, ii, wnew]
#
# (the left operator environment `Lenv` and the on-site MPO `W`, applied to the
# site tensor `A = psi[p]`), then widens the right bond of `A[p]` by direct-summing
# `alpha * P` onto it and zero-pads the left bond of `A[p+1]` to match.  Because the
# padding block of `A[p+1]` is zero, the represented state is UNCHANGED — only the
# bond's column space is enriched, giving the subsequent SVD new directions to keep.
# The strength `alpha` (small, decayed across sweeps) is supplied by the caller;
# this routine is pure structure and applies no truncation.
#
# Leg-order conventions (as everywhere in this project):
#   MPS tensor : Array{T,3}  (l, i, r)
#   MPO tensor : Array{T,4}  (l, ip, i, r)    ip = outgoing physical, i = incoming
#   Op env     : Array{T,3}  (dn, w, up)      dn = ket virtual, w = MPO, up = bra
#
# The direction ("right"/"left") selects which bond of the pair is grown and which
# environment/MPO drives the expansion, so the same routine serves both halves of a
# sweep.  Everything is written with `@tensor` and `direct_sum`, so it is agnostic
# to the element type (Float64 / ComplexF64 both work).

using TensorOperations

"""
    expansion_term(env, A, W, direction) -> P

The 3S expansion tensor at one site, applied to site tensor `A[l, i, r]` with
on-site MPO `W[wl, ip, i, wr]` and the operator environment `env` on the side the
bond is grown from:

- `direction = "right"`: `env` is the LEFT environment `Lenv[dn, w, up]` (its `up`
  leg is the ket bond `l` of `A`); returns `P[l, i, r*wr]` — the outgoing MPO bond
  `wr` is flattened into the right bond, widening it by a factor of `size(W, 4)`.
- `direction = "left"`:  `env` is the RIGHT environment `Renv[dn, w, up]` (its `up`
  leg is the ket bond `r` of `A`); returns `P[l*wl, i, r]` — the incoming MPO bond
  `wl` is flattened into the left bond.

`P` is the raw expansion term; the caller scales it by the expansion strength and
direct-sums it onto `A` (see [`expand_bond`](@ref)).
"""
function expansion_term(env::Array{T,3}, A::Array{T,3}, W::Array{T,4},
                        direction::AbstractString) where {T<:Number}
    if direction == "right"
        # Grow the RIGHT bond of A using the LEFT environment.
        # env = Lenv[dn, w, up]; contract its ket leg (dn) with A's left bond and
        # its MPO leg (w) with W's left MPO leg; keep up as the new left bond `l`.
        @tensor P4[l, i, r, wr] :=
            env[dn, w, l] * A[dn, ii, r] * W[w, i, ii, wr]
        lb, pd, rb, wb = size(P4)
        return reshape(P4, lb, pd, rb * wb)
    elseif direction == "left"
        # Grow the LEFT bond of A using the RIGHT environment.
        # env = Renv[dn, w, up]; contract its ket leg (dn) with A's right bond and
        # its MPO leg (w) with W's right MPO leg; keep up as the new right bond `r`.
        @tensor P4[l, wl, i, r] :=
            env[dn, w, r] * A[l, ii, dn] * W[wl, i, ii, w]
        lb, wb, pd, rb = size(P4)
        return reshape(P4, lb * wb, pd, rb)
    else
        error("expansion_term: direction must be \"right\" or \"left\"; got \"$direction\".")
    end
end

"""
    expand_bond(Ap, Anext, env, W, direction; alpha) -> (Ap_exp, Anext_exp)

3S subspace expansion of the bond BETWEEN `Ap` and `Anext` (both `Array{T,3}` MPS
site tensors), returning widened tensors that represent the SAME state.

- `direction = "right"` (rightward move, center at `p` going to `p+1`): `env` is
  the left operator environment `Lenv` at bond `p-1`, `W` the on-site MPO at `p`.
  The shared bond (right of `Ap`, left of `Anext`) is grown: `Ap` gets
  `alpha * expansion_term(...)` direct-summed onto its right bond, and `Anext` is
  zero-padded on its left bond by the same amount.
- `direction = "left"` (leftward move): `env` is the right operator environment
  `Renv` at bond `p+1`, `W` the on-site MPO at `p` (here `Ap` is the site being
  expanded and `Anext` is its LEFT neighbour).  The shared bond (left of `Ap`,
  right of `Anext`) is grown symmetrically.

`alpha` is the expansion strength (0 recovers the inputs unchanged, up to the
extra zero block).  No truncation is applied — the caller splits/truncates the
widened tensors afterwards.  Because the added block of `Anext` is zero, the
contraction `Ap * Anext` is unchanged, so the state is preserved exactly.
"""
function expand_bond(Ap::Array{T,3}, Anext::Array{T,3},
                     env::Array{T,3}, W::Array{T,4},
                     direction::AbstractString; alpha::Real=1.0) where {T<:Number}
    P = expansion_term(env, Ap, W, direction)

    if direction == "right"
        size(P, 1) == size(Ap, 1) && size(P, 2) == size(Ap, 2) ||
            error("expand_bond: right expansion term shape $(size(P)) incompatible " *
                  "with Ap $(size(Ap)).")
        # Widen Ap's right bond (dim 3) with alpha*P.
        Ap_exp = direct_sum(Ap, T(alpha) .* P, [3])
        # Zero-pad Anext's left bond (dim 1) by the width of P's new right bond.
        pad = size(P, 3)
        Z = zeros(T, pad, size(Anext, 2), size(Anext, 3))
        Anext_exp = direct_sum(Anext, Z, [1])
        return Ap_exp, Anext_exp
    else  # "left"
        size(P, 2) == size(Ap, 2) && size(P, 3) == size(Ap, 3) ||
            error("expand_bond: left expansion term shape $(size(P)) incompatible " *
                  "with Ap $(size(Ap)).")
        # Widen Ap's left bond (dim 1) with alpha*P.
        Ap_exp = direct_sum(Ap, T(alpha) .* P, [1])
        # Zero-pad Anext's right bond (dim 3) by the width of P's new left bond.
        pad = size(P, 1)
        Z = zeros(T, size(Anext, 1), size(Anext, 2), pad)
        Anext_exp = direct_sum(Anext, Z, [3])
        return Ap_exp, Anext_exp
    end
end
