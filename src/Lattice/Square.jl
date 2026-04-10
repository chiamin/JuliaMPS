# Square.jl
#
# Square lattice with row-major 1D site mapping.
#
# Coordinates are 1-indexed (Julia convention):
#   x ∈ 1..Lx  (column),  y ∈ 1..Ly  (row)
#   site_index(x, y) = (y - 1) * Lx + x   ∈ 1..Lx*Ly
#
# Bonds are sorted (i, j) pairs with i < j, also 1-indexed.

"""
    SquareLattice

2D square lattice mapped to a 1D chain via row-major (C) ordering.
All coordinates and site indices are 1-indexed.
"""
struct SquareLattice
    Lx   :: Int
    Ly   :: Int
    xpbc :: Bool
    ypbc :: Bool
    _bonds :: Vector{Tuple{Int,Int}}  # precomputed, sorted, i < j, 1-indexed
end

"""
    SquareLattice(Lx, Ly; xpbc=false, ypbc=false) -> SquareLattice

Construct a square lattice of size Lx×Ly with optional periodic boundaries.
"""
function SquareLattice(Lx::Int, Ly::Int; xpbc::Bool=false, ypbc::Bool=false)
    (Lx >= 1 && Ly >= 1) || error("Lx and Ly must be >= 1; got Lx=$Lx, Ly=$Ly.")
    bonds = _build_bonds(Lx, Ly, xpbc, ypbc)
    return SquareLattice(Lx, Ly, xpbc, ypbc, bonds)
end

# Number of sites
Base.length(lat::SquareLattice) = lat.Lx * lat.Ly

"""
    site_index(lat, x, y) -> Int

Return the 1-indexed 1D site index for lattice coordinate (x, y).
Both `x` and `y` are 1-indexed: x ∈ 1..Lx, y ∈ 1..Ly.
"""
site_index(lat::SquareLattice, x::Int, y::Int) = (y - 1) * lat.Lx + x

"""
    site_coord(lat, i) -> (x, y)

Return the 1-indexed 2D coordinate `(x, y)` for site `i ∈ 1..Lx*Ly`.
"""
function site_coord(lat::SquareLattice, i::Int)
    q, r = divrem(i - 1, lat.Lx)
    return (r + 1, q + 1)
end

"""
    bonds(lat) -> Vector{Tuple{Int,Int}}

Return all nearest-neighbor bonds as sorted `(i, j)` pairs with `i < j`,
1-indexed.
"""
bonds(lat::SquareLattice) = lat._bonds

function _build_bonds(Lx, Ly, xpbc, ypbc)
    bond_set = Set{Tuple{Int,Int}}()
    @inline idx(x, y) = (y - 1) * Lx + x
    for y in 1:Ly, x in 1:Lx
        i = idx(x, y)
        # x-direction
        if x + 1 <= Lx
            push!(bond_set, (i, idx(x + 1, y)))
        elseif xpbc && Lx > 1
            j = idx(1, y)
            push!(bond_set, (min(i, j), max(i, j)))
        end
        # y-direction
        if y + 1 <= Ly
            push!(bond_set, (i, idx(x, y + 1)))
        elseif ypbc && Ly > 1
            j = idx(x, 1)
            push!(bond_set, (min(i, j), max(i, j)))
        end
    end
    return sort!(collect(bond_set))
end

function Base.show(io::IO, lat::SquareLattice)
    bc = String[]
    lat.xpbc && push!(bc, "xpbc")
    lat.ypbc && push!(bc, "ypbc")
    bc_str = isempty(bc) ? "" : ", bc=$(join(bc, '+'))"
    print(io, "SquareLattice(Lx=$(lat.Lx), Ly=$(lat.Ly)$bc_str)")
end
