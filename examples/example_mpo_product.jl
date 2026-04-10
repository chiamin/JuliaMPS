# example_mpo_product.jl
#
# MPO product and compression example: H² for the spin-1/2 Heisenberg chain.
#
# Demonstrates `mpo_product` (exact product, bond dimensions multiply) and
# `svd_compress_mpo` (SVD truncation).  We then validate by computing
# ⟨ψ|H²|ψ⟩ on a few product states with both the exact and the compressed
# H², and checking that the difference is below the cutoff.
#
# Note: `fit_mpo_product` from the original Python project is not implemented
# in MPSCore (it is left as a stub).

using MPSCore
using LinearAlgebra

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

const N       = 10
const MAX_DIM = 12      # target bond dimension for the compressed H²


# ---------------------------------------------------------------------------
# Build the Heisenberg Hamiltonian
# ---------------------------------------------------------------------------

site = spin_half()

ampo = AutoMPO(N, site)
for i in 1:N-1
    add!(ampo, 1.0, "Sz", i, "Sz", i+1)
    add!(ampo, 0.5, "Sp", i, "Sm", i+1)
    add!(ampo, 0.5, "Sm", i, "Sp", i+1)
end
H = to_mpo(ampo)

println("H bond dims:        ", mpo_dims(H))


# ---------------------------------------------------------------------------
# Exact H² and SVD-compressed H²
# ---------------------------------------------------------------------------

H2_exact = mpo_product(H, H)
println("H² exact bond dims: ", mpo_dims(H2_exact))

H2_svd = svd_compress_mpo(H2_exact; max_dim=MAX_DIM, cutoff=1e-14)
println("H² SVD compressed:  ", mpo_dims(H2_svd))


# ---------------------------------------------------------------------------
# Validate against a few product states
# ---------------------------------------------------------------------------

up = [1.0, 0.0]
dn = [0.0, 1.0]

states_list = [
    [iseven(k) ? dn : up for k in 1:N],                # Néel ↑↓↑↓...
    [iseven(k) ? up : dn for k in 1:N],                # reversed Néel
    [k <= div(N, 2) ? up : dn for k in 1:N],           # domain wall ↑↑↑↓↓↓
]

println()
println("Validation: ⟨ψ|H²|ψ⟩ on a few product states")
println(rpad("state", 18),
        lpad("exact", 18), lpad("SVD", 18), lpad("|err|", 12))
println("-" ^ 66)

errors = Float64[]
for states in states_list
    psi = product_state(Float64, states)
    orthogonalize!(psi)
    LinearAlgebra.normalize!(psi)
    v_exact = real(expectation(psi, H2_exact, psi))
    v_svd   = real(expectation(psi, H2_svd,   psi))
    err     = abs(v_exact - v_svd)
    push!(errors, err)
    label = join([s == up ? "↑" : "↓" for s in states])
    println(rpad(label, 18),
            lpad(string(round(v_exact; digits=10)), 18),
            lpad(string(round(v_svd;   digits=10)), 18),
            lpad(string(round(err; sigdigits=2)), 12))
end

println()
println("Maximum compression error: ", round(maximum(errors); sigdigits=2))
