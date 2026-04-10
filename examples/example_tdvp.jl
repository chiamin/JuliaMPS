# example_tdvp.jl
#
# Real-time TDVP example: spin-1/2 Heisenberg chain.
#
# Starts from a Néel product state and evolves with H = J Σ S_i · S_{i+1}
# using complex tensors and a small real-time step  dt = i Δt.
# Both energy and norm should be conserved to high accuracy.

using MPSCore
using LinearAlgebra

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

const N       = 8
const J       = 1.0
const Δt      = 0.1
const NSTEPS  = 20
const MAX_DIM = 32


# ---------------------------------------------------------------------------
# Build Heisenberg MPO (complex)
# ---------------------------------------------------------------------------

function build_heisenberg(N::Int, J::Float64)
    site = spin_half()
    ampo = AutoMPO(N, site)
    for i in 1:N-1
        add!(ampo, J,   "Sz", i, "Sz", i+1)
        add!(ampo, J/2, "Sp", i, "Sm", i+1)
        add!(ampo, J/2, "Sm", i, "Sp", i+1)
    end
    H_real = to_mpo(ampo)
    # All operators here are real, so AutoMPO returns a Float64 MPO.
    # Convert to ComplexF64 so it can pair with a complex MPS in TDVPEngine.
    return MPO([ComplexF64.(H_real[k]) for k in 1:N])
end


# ---------------------------------------------------------------------------
# Initial Néel state in ComplexF64
# ---------------------------------------------------------------------------

function neel_state(N::Int)
    up = ComplexF64[1.0, 0.0]
    dn = ComplexF64[0.0, 1.0]
    states = [iseven(k) ? dn : up for k in 1:N]
    psi = product_state(ComplexF64, states)
    orthogonalize!(psi)
    return psi
end


# ---------------------------------------------------------------------------
# Run real-time TDVP
# ---------------------------------------------------------------------------

H   = build_heisenberg(N, J)
psi = neel_state(N)

E0  = real(expectation(psi, H, psi))
n0  = norm(psi)

println("=" ^ 56)
println("  Real-time TDVP: Heisenberg chain (N = $N)")
println("=" ^ 56)
println("  initial E = ", round(E0; digits=8))
println("  initial ‖ψ‖ = ", round(n0; digits=8))
println()

println("step       E              ‖ψ‖             trunc")
println("-" ^ 56)

engine = TDVPEngine(psi, H)
for step in 1:NSTEPS
    trunc = sweep!(engine, 1im * Δt;
                   num_center=2, max_dim=MAX_DIM, cutoff=1e-12)
    E = real(expectation(psi, H, psi))
    nrm = norm(psi)
    println(rpad(string(step), 5),
            lpad(string(round(E; digits=8)), 16),
            lpad(string(round(nrm; digits=8)), 16),
            lpad(string(round(trunc; sigdigits=2)), 14))
end

println()
println("Energy drift  : ", round(real(expectation(psi, H, psi)) - E0; sigdigits=3))
println("Norm   drift  : ", round(norm(psi) - n0; sigdigits=3))
