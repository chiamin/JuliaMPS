# AutoMPO.jl
#
# Construct general MPOs from sums of operator strings via the
# finite-state-machine (FSM) algorithm with prefix merging.  Dense-only.
#
# Usage:
#
#     site = spin_half()
#     ampo = AutoMPO(N, site)
#     for i in 1:N-1
#         add!(ampo, J/2, "Sp", i, "Sm", i+1)
#         add!(ampo, J/2, "Sm", i, "Sp", i+1)
#         add!(ampo, J,   "Sz", i, "Sz", i+1)
#     end
#     H = to_mpo(ampo)
#
# Sites are 1-indexed (Julia convention).
#
# Jordan-Wigner strings are inserted automatically for fermionic operators
# (those marked `fermionic=true` in the PhysicalSite registry).

# ---------------------------------------------------------------------------
# Internal data structures
# ---------------------------------------------------------------------------

"""
    OpEntry

A single per-site operator after JW expansion:
  - `site`    : 1-indexed site index
  - `sym_key` : sequence of operator names applied at this site (e.g. ["Cdag","F"])
  - `matrix`  : the resulting d×d matrix
"""
struct OpEntry
    site    :: Int
    sym_key :: Vector{String}
    matrix  :: Matrix{ComplexF64}
end

"""
    Term

One term `coeff * O_{s1} O_{s2} ... O_{sk}` after JW preprocessing.
Operators are sorted by site index.
"""
struct Term
    coeff :: ComplexF64
    ops   :: Vector{OpEntry}
end

# ---------------------------------------------------------------------------
# AutoMPO
# ---------------------------------------------------------------------------

mutable struct AutoMPO
    N     :: Int
    site  :: PhysicalSite
    terms :: Vector{Term}
end

"""
    AutoMPO(N, site) -> AutoMPO

Create an empty AutoMPO for a chain of `N` identical sites.
"""
function AutoMPO(N::Int, site::PhysicalSite)
    N >= 1 || error("N must be >= 1.")
    return AutoMPO(N, site, Term[])
end

"""
    add!(ampo, coeff, "Op1", site1, "Op2", site2, ...)

Add one term to the Hamiltonian.  Operators may be in any site order; the
JW expansion is computed automatically for fermionic operators.

Sites are 1-indexed.
"""
function add!(ampo::AutoMPO, coeff::Number, args...)
    iseven(length(args)) ||
        error("add!() expects pairs (op_name, site); got $(length(args)) extra arg(s).")
    user_ops = Tuple{Int,String}[]
    for k in 1:2:length(args)
        op_name = args[k]
        site_idx = args[k+1]
        op_name isa AbstractString ||
            error("Expected operator name (String); got $(typeof(op_name)).")
        site_idx isa Integer ||
            error("Expected site index (Int); got $(typeof(site_idx)).")
        (1 <= site_idx <= ampo.N) ||
            error("Site $site_idx out of range [1, $(ampo.N)].")
        op(ampo.site, String(op_name))   # validate name exists
        push!(user_ops, (Int(site_idx), String(op_name)))
    end
    processed = _preprocess_term(ampo.site, user_ops)
    push!(ampo.terms, Term(ComplexF64(coeff), processed))
    return ampo
end

# ---------------------------------------------------------------------------
# JW preprocessing
# ---------------------------------------------------------------------------

"""
Expand JW strings and compute on-site operators for one term.
For each site `s` that has a user operator, walk through the term left-to-right:
  - if op acts on site s: append its name
  - if op acts on a site > s and is fermionic: append "F" (cancels with prev F)
The resulting per-site operator matrix is the product of the symbolic sequence.
Returns a list of OpEntry sorted by site.
"""
function _preprocess_term(site_obj::PhysicalSite,
                          user_ops::Vector{Tuple{Int,String}})
    # Unique sites that have user operators (sorted)
    operator_sites = sort!(unique(s for (s, _) in user_ops))
    result = OpEntry[]
    d = site_obj.dim
    for s in operator_sites
        seq = String[]
        for (s_k, op_name) in user_ops
            if s_k == s
                push!(seq, op_name)
            elseif s_k > s && is_fermionic(site_obj, op_name)
                if !isempty(seq) && seq[end] == "F"
                    pop!(seq)            # F² = I, cancel
                else
                    push!(seq, "F")
                end
            end
        end
        # Multiply matrices left-to-right
        mat = Matrix{ComplexF64}(I, d, d)
        for name in seq
            mat = mat * op(site_obj, name)
        end
        push!(result, OpEntry(s, seq, mat))
    end
    return result
end

# ---------------------------------------------------------------------------
# FSM build
# ---------------------------------------------------------------------------

# State key: either :start, :done, or a tuple-of-tuples representing the partial.
# Use Any for heterogeneous storage; sort/lookup by `string(key)`.

const _DONE  = :done
const _START = :start

"""
Build an immutable, hashable representation of a partial key from a slice of
ops.  Uses `Tuple{Int, NTuple{M,String}}` per entry so the result is hashable.
"""
function _make_partial_key(ops::Vector{OpEntry}, j::Int)
    return Tuple((e.site, Tuple(e.sym_key)) for e in ops[1:j])
end

"""
Return true if the partial key has applied an odd number of fermionic operators.
"""
function _partial_is_fermionic(site_obj::PhysicalSite, key)
    count = 0
    for (_, sym_key) in key
        for name in sym_key
            if name != "F" && is_fermionic(site_obj, name)
                count += 1
            end
        end
    end
    return isodd(count)
end

"""
Enumerate the FSM states at every bond.

bond_states[p] (1-indexed, p ∈ 1..N+1):
  - bond p=1     : left boundary, contains only :start
  - bond p=N+1   : right boundary, contains only :done
  - interior     : :done, :start, plus all in-progress partial keys
                    from terms that cross this bond.
"""
function _enumerate_states(ampo::AutoMPO)
    N = ampo.N
    in_progress = [Set{Any}() for _ in 1:N+1]
    for term in ampo.terms
        ops = term.ops
        for j in 1:length(ops)-1
            partial_key = _make_partial_key(ops, j)
            left_site  = ops[j].site
            right_site = ops[j+1].site
            # Active at bonds (left_site+1) .. right_site (1-indexed bond between sites)
            # Bond between site k and site k+1 has bond index k+1 (1-indexed); the
            # bond before site 1 is index 1, and the bond after site N is index N+1.
            for p in (left_site+1):right_site
                push!(in_progress[p], partial_key)
            end
        end
    end

    bond_states = Vector{Vector{Any}}(undef, N+1)
    for p in 1:N+1
        partials = sort!(collect(in_progress[p]); by=string)
        all_states = Any[_DONE; partials; _START]
        bond_states[p] = all_states
    end
    bond_states[1]   = Any[_START]
    bond_states[N+1] = Any[_DONE]
    return bond_states
end

# Find the index of `key` in `states`, or `nothing` if absent.
function _state_index(states::Vector{Any}, key)
    for (i, s) in enumerate(states)
        s === key && return i
        s == key  && return i
    end
    return nothing
end

"""
Decide the global element type for the MPO from coefficient and matrix dtypes.
"""
function _resolve_dtype(ampo::AutoMPO)
    is_complex = false
    # Operator matrices in PhysicalSite are stored as ComplexF64; check imag parts
    for term in ampo.terms
        if !iszero(imag(term.coeff))
            is_complex = true; break
        end
        for e in term.ops
            if any(!iszero, imag.(e.matrix))
                is_complex = true; break
            end
        end
        is_complex && break
    end
    return is_complex ? ComplexF64 : Float64
end

"""
Fill the diagonal "pass-through" entries of W:
  - DONE → DONE, START → START : identity I
  - in-progress partial → same partial : F if odd fermion count, else I
"""
function _fill_identity!(W::Array{T,4},
                         site_obj::PhysicalSite,
                         left_states::Vector{Any},
                         right_states::Vector{Any}) where {T}
    I_mat = T.(op(site_obj, "I"))
    F_mat = haskey(site_obj.ops, "F") ? T.(op(site_obj, "F")) : nothing
    # DONE → DONE, START → START
    for key in (_DONE, _START)
        l = _state_index(left_states, key)
        r = _state_index(right_states, key)
        if l !== nothing && r !== nothing
            @views W[l, :, :, r] .+= I_mat
        end
    end
    # Partials → same partial
    for key in left_states
        (key === _DONE || key === _START) && continue
        l = _state_index(left_states, key)
        r = _state_index(right_states, key)
        if l !== nothing && r !== nothing
            if _partial_is_fermionic(site_obj, key)
                F_mat === nothing &&
                    error("Fermionic operator without 'F' registered on the site.")
                @views W[l, :, :, r] .+= F_mat
            else
                @views W[l, :, :, r] .+= I_mat
            end
        end
    end
end

"""
Add one term's contributions to W at site p.

The coefficient is placed on the LAST operator of each term so that
shared-prefix transitions carry only the operator matrix.
"""
function _fill_term!(W::Array{T,4},
                     p::Int,
                     term::Term,
                     left_states::Vector{Any},
                     right_states::Vector{Any},
                     seen::Set{Tuple{Int,Int}}) where {T}
    ops = term.ops
    # Find which entries of `ops` act at site p
    acting = [(j, e) for (j, e) in enumerate(ops) if e.site == p]
    isempty(acting) && return
    for (j, e) in acting
        partial_left  = _make_partial_key(ops, j-1)   # ops[1..j-1]
        partial_right = _make_partial_key(ops, j)     # ops[1..j]

        l_key = j == 1            ? _START : partial_left
        is_terminal = j == length(ops)
        r_key = is_terminal       ? _DONE  : partial_right

        l = _state_index(left_states, l_key)
        r = _state_index(right_states, r_key)
        (l === nothing || r === nothing) && continue

        if is_terminal
            @views W[l, :, :, r] .+= T(term.coeff) .* T.(e.matrix)
        else
            lr = (l, r)
            if !(lr in seen)
                push!(seen, lr)
                @views W[l, :, :, r] .+= T.(e.matrix)
            end
        end
    end
end

"""
    to_mpo(ampo) -> MPO

Build and return the MPO from the accumulated terms.
"""
function to_mpo(ampo::AutoMPO)
    isempty(ampo.terms) && error("AutoMPO has no terms.")
    N = ampo.N
    site_obj = ampo.site
    d = site_obj.dim
    T = _resolve_dtype(ampo)

    bond_states = _enumerate_states(ampo)

    tensors = Vector{Array{T,4}}(undef, N)
    for p in 1:N
        left_states  = bond_states[p]
        right_states = bond_states[p+1]
        Dl = length(left_states)
        Dr = length(right_states)
        # Leg order (l, ip, i, r); ip = row of operator matrix, i = column
        W = zeros(T, Dl, d, d, Dr)
        _fill_identity!(W, site_obj, left_states, right_states)
        seen = Set{Tuple{Int,Int}}()
        for term in ampo.terms
            _fill_term!(W, p, term, left_states, right_states, seen)
        end
        tensors[p] = W
    end
    return MPO(tensors)
end
