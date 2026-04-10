# example_fermion_2d.jl
#
# 2D spinless-fermion tight-binding model on a 3×3 square lattice:
#
#   H = -t Σ_{<i,j>} (c†_i c_j + c†_j c_i)
#
# Builds the MPO via AutoMPO with automatic Jordan-Wigner strings, then
# verifies the result against direct exact diagonalisation in the
# occupation-number basis.
#
# Single-particle energies for OBC are
#   ε_{nx,ny} = -2t [cos(π nx/(Lx+1)) + cos(π ny/(Ly+1))]
# and the many-body ground state fills the lowest single-particle levels.

using MPSCore
using LinearAlgebra
using TensorOperations

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

const Lx, Ly = 3, 3
const t = 1.0

lat = SquareLattice(Lx, Ly)
const N = length(lat)
const dim = 2^N

println("Lattice: ", lat)
println("N = $N sites,  Hilbert space dim = $dim")
println("Bonds: ", bonds(lat))
println()


# ---------------------------------------------------------------------------
# Build MPO via AutoMPO
# ---------------------------------------------------------------------------

site = spinless_fermion()
ampo = AutoMPO(N, site)
for (i, j) in bonds(lat)
    add!(ampo, -t, "Cdag", i, "C", j)
    add!(ampo, -t, "Cdag", j, "C", i)
end
H = to_mpo(ampo)

println("MPO bond dimensions: ", mpo_dims(H))
println()


# ---------------------------------------------------------------------------
# MPO -> dense matrix
# ---------------------------------------------------------------------------

function mpo_to_matrix(mpo::MPO{T}) where {T}
    Nsites = length(mpo)
    d = size(mpo[1], 2)
    block = copy(mpo[1])
    for k in 2:Nsites
        Wk = mpo[k]
        sz_b = size(block); sz_w = size(Wk)
        bmat = reshape(block, prod(sz_b[1:end-1]), sz_b[end])
        wmat = reshape(Wk,    sz_w[1], sz_w[2]*sz_w[3]*sz_w[4])
        cmat = bmat * wmat
        block = reshape(cmat, sz_b[1:end-1]..., sz_w[2], sz_w[3], sz_w[4])
    end
    ip_dims = [2*k     for k in 1:Nsites]
    i_dims  = [2*k + 1 for k in 1:Nsites]
    perm    = [reverse(ip_dims); reverse(i_dims); 1; 2*Nsites + 2]
    block_p = permutedims(block, perm)
    return reshape(block_p, d^Nsites, d^Nsites)
end


# ---------------------------------------------------------------------------
# Direct ED with explicit Jordan-Wigner
# ---------------------------------------------------------------------------

function tb_ed_matrix(lat::SquareLattice, t::Float64)
    Nsites = length(lat)
    dim = 2^Nsites
    H = zeros(dim, dim)
    bond_list = bonds(lat)
    for state in 0:dim-1
        # bit k (1..N) = 1 iff site k is occupied
        bits = [(state >> (Nsites - k)) & 1 for k in 1:Nsites]
        for (i, j) in bond_list
            if bits[j] == 1 && bits[i] == 0
                new_bits = copy(bits)
                new_bits[j] = 0
                new_bits[i] = 1
                new_state = sum(new_bits[k] << (Nsites - k) for k in 1:Nsites)
                # JW sign: parity of fermions strictly between i and j
                sign = (-1)^sum(bits[k] for k in (i+1):(j-1); init=0)
                H[new_state + 1, state + 1] += -t * sign
                H[state + 1, new_state + 1] += -t * sign
            end
        end
    end
    return H
end


H_mpo_mat = mpo_to_matrix(H)
H_ed      = tb_ed_matrix(lat, t)

max_diff = maximum(abs.(H_mpo_mat .- H_ed))
println("Max |H_mpo - H_ed| = ", max_diff)
@assert max_diff < 1e-12 "MPO and ED matrices differ!"
println("MPO matches exact diagonalization. ✓")
println()


# ---------------------------------------------------------------------------
# Spectrum vs single-particle energies
# ---------------------------------------------------------------------------

evals = sort(real.(eigvals(Hermitian(H_mpo_mat))))

sp_energies = sort([
    -2.0 * t * (cos(π * nx / (Lx + 1)) + cos(π * ny / (Ly + 1)))
    for nx in 1:Lx, ny in 1:Ly
] |> vec)

println("Single-particle energies:")
for (k, e) in enumerate(sp_energies)
    println("  ε_$(k-1) = ", round(e; digits=6))
end
println()

# Many-body ground state: try every filling, pick the minimum
E0_sp_min, Nf_best = findmin([sum(sp_energies[1:Nf]) for Nf in 0:N])
Nf_best -= 1   # convert from 1-indexed to filling number

println("Many-body ground state (filling $Nf_best particles):")
println("  E0 (from single-particle) = ", round(E0_sp_min; digits=8))
println("  E0 (from MPO eigvalsh)    = ", round(evals[1]; digits=8))
println("  Difference                = ", round(abs(evals[1] - E0_sp_min); sigdigits=2))
