# example_product_state.jl
#
# Product-state MPS example.
#
# Build a Néel product state |↑↓↑↓...> on a spin-1/2 chain and verify its
# norm and bond dimensions.  Spin-1/2 basis (MPSCore convention):
#     index 1 = |↑⟩  (Sz = +1/2)
#     index 2 = |↓⟩  (Sz = -1/2)

using MPSCore
using LinearAlgebra

N = 6

# Néel state: alternating up / down (length N)
# Basis convention: 1 = |↑⟩, 2 = |↓⟩.  Each entry is a 2-vector.
up = [1.0, 0.0]   # |↑⟩
dn = [0.0, 1.0]   # |↓⟩
neel_states = [iseven(k) ? dn : up for k in 1:N]

psi = product_state(Float64, neel_states)

println("Product state |↑↓↑↓...⟩, N = $N")
println("  num sites  = ", length(psi))
println("  bond dims  = ", bond_dims(psi))
println("  ⟨ψ|ψ⟩      = ", real(inner(psi, psi)))
println("  norm       = ", norm(psi))

# Sanity: <psi|Sz_k|psi> = (-1)^(k-1) * 0.5 for the Neel state
site = spin_half()
println("\nLocal magnetisation ⟨Sz_k⟩:")
for k in 1:N
    # Build a 1-site MPO that is identity everywhere except Sz at site k
    ampo = AutoMPO(N, site)
    add!(ampo, 1.0, "Sz", k)
    Mk = to_mpo(ampo)
    sz_k = real(expectation(psi, Mk, psi))
    println("  k = $k :  ⟨Sz_$k⟩ = $(round(sz_k; digits=6))")
end
