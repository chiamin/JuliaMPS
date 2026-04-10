# example_hubbard.jl
#
# 1D Hubbard model on N=4 sites:
#
#   H = -t Σ_{i,σ} (c†_{i,σ} c_{i+1,σ} + h.c.)  +  U Σ_i n_{i↑} n_{i↓}
#
# Builds the MPO via AutoMPO (with automatic Jordan-Wigner strings) and
# compares the dense MPO matrix against direct exact diagonalisation.

using MPSCore
using LinearAlgebra
using TensorOperations

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

const N = 4
const t = 1.0
const U = 4.0

println("1D Hubbard chain: N=$N, t=$t, U=$U")
println("Hilbert space dim = 4^$N = $(4^N)")
println()


# ---------------------------------------------------------------------------
# Build MPO via AutoMPO
# ---------------------------------------------------------------------------

site = electron()
ampo = AutoMPO(N, site)
for i in 1:N-1
    for (op_dag, op_) in [("Cupdag", "Cup"), ("Cdndag", "Cdn")]
        add!(ampo, -t, op_dag, i,   op_, i+1)
        add!(ampo, -t, op_dag, i+1, op_, i)
    end
end
for i in 1:N
    add!(ampo, U, "Nup", i, "Ndn", i)
end
H = to_mpo(ampo)

println("MPO bond dimensions: ", mpo_dims(H))
println()


# ---------------------------------------------------------------------------
# MPO -> dense matrix (kron-ordered, ip slowest)
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
# Direct ED (Hubbard in occupation-number basis with explicit JW)
# ---------------------------------------------------------------------------

# Local basis index k (1..4) ↔ (n↑, n↓) bit pattern,
# matching the Electron site convention:
#   1 = |0⟩,  2 = |↑⟩,  3 = |↓⟩,  4 = |↑↓⟩
#   |↑↓⟩ = c†_↑ c†_↓ |0⟩  (c†_↓ acts first)
const ELECTRON_NUP = (0, 1, 0, 1)
const ELECTRON_NDN = (0, 0, 1, 1)

function build_hubbard_ed(N::Int, t::Float64, U::Float64)
    Cup    = real(op(electron(), "Cup"))
    Cupdag = real(op(electron(), "Cupdag"))
    Cdn    = real(op(electron(), "Cdn"))
    Cdndag = real(op(electron(), "Cdndag"))
    Nup    = real(op(electron(), "Nup"))
    Ndn    = real(op(electron(), "Ndn"))
    F      = real(op(electron(), "F"))
    Id     = Matrix{Float64}(I, 4, 4)

    # Build the operator at site k using JW: F at all sites < k, op at k.
    function fermion_op(k::Int, local_op)
        out = (k == 1) ? local_op : F
        for j in 2:N
            piece = j < k ? F : (j == k ? local_op : Id)
            out = kron(out, piece)
        end
        return out
    end
    function on_site_op(k::Int, local_op)
        out = (k == 1) ? local_op : Id
        for j in 2:N
            out = kron(out, j == k ? local_op : Id)
        end
        return out
    end

    H = zeros(4^N, 4^N)
    for i in 1:N-1
        # spin up hopping
        H .+= -t .* fermion_op(i,   Cupdag) * fermion_op(i+1, Cup)
        H .+= -t .* fermion_op(i+1, Cupdag) * fermion_op(i,   Cup)
        # spin down hopping
        H .+= -t .* fermion_op(i,   Cdndag) * fermion_op(i+1, Cdn)
        H .+= -t .* fermion_op(i+1, Cdndag) * fermion_op(i,   Cdn)
    end
    for i in 1:N
        H .+= U .* on_site_op(i, Nup) * on_site_op(i, Ndn)
    end
    return H
end


H_mpo = mpo_to_matrix(H)
H_ed  = build_hubbard_ed(N, t, U)

max_diff = maximum(abs.(H_mpo .- H_ed))
println("Max |H_mpo - H_ed| = ", max_diff)
@assert max_diff < 1e-12 "MPO and ED matrices differ!"
println("MPO matches exact diagonalization. ✓")
println()


# ---------------------------------------------------------------------------
# Spectrum
# ---------------------------------------------------------------------------

evals = sort(real.(eigvals(Hermitian(H_mpo))))
println("Ground state energy:  E0 = ", round(evals[1]; digits=8))
println("First gap:            ΔE = ", round(evals[2] - evals[1]; digits=8))
println()
println("Lowest 10 eigenvalues:")
for k in 1:min(10, length(evals))
    println("  E_$(k-1) = ", round(evals[k]; digits=8))
end
