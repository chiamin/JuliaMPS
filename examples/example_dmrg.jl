# example_dmrg.jl
#
# DMRG ground-state example: spin-1/2 Heisenberg chain.
#
#     H = J Σ_i [ Sz_i Sz_{i+1} + (S+_i S-_{i+1} + S-_i S+_{i+1}) / 2 ]
#
# The exact ground-state energy per site for the isotropic chain in the
# thermodynamic limit is given by the Bethe ansatz:
#     E0/N → J (1/4 - ln 2) ≈ -0.4431 J

using MPSCore
using Random

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

const N     = 10      # number of sites
const J     = 1.0     # exchange coupling
const seed  = 42

Random.seed!(seed)

# DMRG schedule: per-sweep (max_dim, cutoff)
const SCHEDULE = [
    (10,  0.0),
    (20,  0.0),
    (40,  1e-8),
    (40,  1e-8),
    (40,  1e-8),
]

const BETHE = J * (0.25 - log(2))   # Bethe ansatz, thermodynamic limit


function build_heisenberg(N::Int, J::Float64)
    site = spin_half()
    ampo = AutoMPO(N, site)
    for i in 1:N-1
        add!(ampo, J,     "Sz", i, "Sz", i+1)
        add!(ampo, J/2,   "Sp", i, "Sm", i+1)
        add!(ampo, J/2,   "Sm", i, "Sp", i+1)
    end
    return to_mpo(ampo)
end


function run_dmrg(label::String, psi::MPS, H::MPO, schedule)
    println("=" ^ 65)
    println("  $label")
    println("=" ^ 65)
    println("sweep    max_dim   cutoff             E         E/N        trunc")
    println("-" ^ 65)
    engine = DMRGEngine(psi, H)
    E = 0.0
    trunc = 0.0
    for (idx, (md, cut)) in enumerate(schedule)
        E, trunc = sweep!(engine; max_dim=md, cutoff=cut, num_center=2)
        println(rpad(string(idx), 5),
                lpad(string(md), 9),
                lpad(string(cut), 11),
                lpad(string(round(E; digits=8)), 14),
                lpad(string(round(E / N; digits=6)), 12),
                lpad(string(round(trunc; sigdigits=2)), 12))
    end
    println()
    println("  Final E     = ", round(E; digits=8))
    println("  Final E/N   = ", round(E / N; digits=8))
    println("  Bethe (TDL) = ", round(BETHE; digits=8))
    return E
end


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

H   = build_heisenberg(N, J)
psi = random_mps(Float64, N, 2, 4)
orthogonalize!(psi)

E = run_dmrg("Heisenberg DMRG (dense, N = $N)", psi, H, SCHEDULE)

println()
println("=" ^ 65)
println("  Summary")
println("=" ^ 65)
println("  E/N         = ", round(E / N; digits=8))
println("  Bethe TDL   = ", round(BETHE; digits=8))
println("  finite-size gap to TDL = ", round(E / N - BETHE; digits=6))
