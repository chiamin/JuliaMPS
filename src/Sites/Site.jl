# Site.jl
#
# Abstract site type and PhysicalSite struct.
# Operators are stored as Matrix{ComplexF64} (row=ip/bra, col=i/ket).

abstract type AbstractSite end

struct PhysicalSite <: AbstractSite
    dim       :: Int
    type_name :: String
    ops       :: Dict{String, Matrix{ComplexF64}}
    fermionic :: Dict{String, Bool}
end

"""
    PhysicalSite(dim, type_name) -> PhysicalSite

Construct an empty PhysicalSite with no operators registered.
Use `register_op!` to add operators.
"""
function PhysicalSite(dim::Int, type_name::String)
    return PhysicalSite(dim, type_name,
                        Dict{String, Matrix{ComplexF64}}(),
                        Dict{String, Bool}())
end

"""
    register_op!(site, name, mat; fermionic=false)

Register a local operator by name.

- `mat`: d×d matrix (row=ip/bra, col=i/ket), any numeric type (converted to ComplexF64).
- `fermionic`: true if the operator has odd fermion parity (used by AutoMPO for JW strings).
"""
function register_op!(site::PhysicalSite, name::String, mat::AbstractMatrix;
                      fermionic::Bool = false)
    d = site.dim
    if size(mat) != (d, d)
        error("Operator '$name' must be ($d,$d); got $(size(mat)).")
    end
    site.ops[name] = ComplexF64.(mat)
    site.fermionic[name] = fermionic
    return site
end

"""
    op(site, name) -> Matrix{ComplexF64}

Return the d×d matrix for operator `name`.
Throws an error with available names if `name` is not registered.
"""
function op(site::PhysicalSite, name::String)
    if !haskey(site.ops, name)
        avail = join(keys(site.ops), ", ")
        error("Operator '$name' not registered in $(site.type_name). Available: $avail")
    end
    return site.ops[name]
end

"""
    is_fermionic(site, name) -> Bool

Return true if operator `name` is fermionic (odd fermion parity).
Returns false if the name is not registered.
"""
function is_fermionic(site::PhysicalSite, name::String)
    return get(site.fermionic, name, false)
end
