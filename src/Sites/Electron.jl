# Electron.jl
#
# Factory function for spin-1/2 electron physical site (Hubbard model).
#
# Basis (fixed, 4 states):
#   index 1 = |0⟩   (empty)           [Python index 0]
#   index 2 = |↑⟩   (spin up)         [Python index 1]
#   index 3 = |↓⟩   (spin down)       [Python index 2]
#   index 4 = |↑↓⟩  (doubly occupied) [Python index 3]
#
# Convention: |↑↓⟩ = c†_↑ c†_↓ |0⟩  (c†_↓ acts first)

"""
    electron() -> PhysicalSite

Create a spin-1/2 electron PhysicalSite for the Hubbard model.

Basis (row/col index in operator matrices):
    1=|0⟩, 2=|↑⟩, 3=|↓⟩, 4=|↑↓⟩

Operators (row=ip/bra, col=i/ket):
    I      = identity (4×4)
    Cup    = c_↑  (annihilate spin-up;  fermionic)
    Cupdag = c†_↑ (create spin-up;      fermionic)
    Cdn    = c_↓  (annihilate spin-down, with sign from c†_↑; fermionic)
    Cdndag = c†_↓ (create spin-down;    fermionic)
    Nup    = n_↑ = c†_↑ c_↑
    Ndn    = n_↓ = c†_↓ c_↓
    Ntot   = n_↑ + n_↓
    Sz     = (n_↑ - n_↓) / 2
    F      = (-1)^Ntot  (Jordan-Wigner parity; non-fermionic)
"""
function electron() :: PhysicalSite
    site = PhysicalSite(4, "Electron")

    I4 = Matrix{Float64}(I, 4, 4)

    # Cup: c_↑ |↑⟩=|0⟩, c_↑ |↑↓⟩=+|↓⟩
    Cup = zeros(Float64, 4, 4)
    Cup[1, 2] = 1.0    # |0⟩ ← |↑⟩
    Cup[3, 4] = 1.0    # |↓⟩ ← |↑↓⟩  (no sign: c_↑ is outermost)

    Cupdag = Cup'  |> Matrix

    # Cdn: c_↓ |↓⟩=|0⟩, c_↓ |↑↓⟩=-|↑⟩  (crosses c†_↑)
    Cdn = zeros(Float64, 4, 4)
    Cdn[1, 3] = 1.0    # |0⟩ ← |↓⟩
    Cdn[2, 4] = -1.0   # |↑⟩ ← |↑↓⟩  (sign from anti-commuting past c†_↑)

    Cdndag = Cdn' |> Matrix

    Nup  = diagm([0.0, 1.0, 0.0, 1.0])
    Ndn  = diagm([0.0, 0.0, 1.0, 1.0])
    Ntot = diagm([0.0, 1.0, 1.0, 2.0])
    Sz   = diagm([0.0, 0.5, -0.5, 0.0])
    F    = diagm([1.0, -1.0, -1.0, 1.0])

    register_op!(site, "I",      I4,     fermionic=false)
    register_op!(site, "Cup",    Cup,    fermionic=true)
    register_op!(site, "Cupdag", Cupdag, fermionic=true)
    register_op!(site, "Cdn",    Cdn,    fermionic=true)
    register_op!(site, "Cdndag", Cdndag, fermionic=true)
    register_op!(site, "Nup",    Nup,    fermionic=false)
    register_op!(site, "Ndn",    Ndn,    fermionic=false)
    register_op!(site, "Ntot",   Ntot,   fermionic=false)
    register_op!(site, "Sz",     Sz,     fermionic=false)
    register_op!(site, "F",      F,      fermionic=false)

    return site
end
