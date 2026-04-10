# SpinlessFermion.jl
#
# Factory function for spinless fermion physical site.
#
# Basis (fixed):
#   index 1 = |0⟩  (empty,    N=0)  [Julia 1-indexed; Python index 0]
#   index 2 = |1⟩  (occupied, N=1)  [Julia 1-indexed; Python index 1]

"""
    spinless_fermion() -> PhysicalSite

Create a spinless fermion PhysicalSite with operators I, N, C, Cdag, F.

Basis:
    index 1 = |0⟩ (empty), index 2 = |1⟩ (occupied)

Operators (row=ip/bra, col=i/ket):
    I    = identity
    N    = number operator: N|1⟩=|1⟩
    C    = annihilation: C|1⟩=|0⟩ → C[1,2]=1  (fermionic)
    Cdag = creation: Cdag|0⟩=|1⟩ → Cdag[2,1]=1  (fermionic)
    F    = parity: (-1)^N = diag(+1,-1)  (non-fermionic)

C and Cdag are fermionic (is_fermionic=true); others are not.
"""
function spinless_fermion() :: PhysicalSite
    site = PhysicalSite(2, "SpinlessFermion")

    I2   = [1.0 0.0; 0.0 1.0]
    N    = [0.0 0.0; 0.0 1.0]
    C    = [0.0 1.0; 0.0 0.0]    # C|1>=|0>: row=empty(1), col=occupied(2)
    Cdag = [0.0 0.0; 1.0 0.0]    # Cdag|0>=|1>: row=occupied(2), col=empty(1)
    F    = [1.0 0.0; 0.0 -1.0]   # (-1)^N

    register_op!(site, "I",    I2,   fermionic=false)
    register_op!(site, "N",    N,    fermionic=false)
    register_op!(site, "C",    C,    fermionic=true)
    register_op!(site, "Cdag", Cdag, fermionic=true)
    register_op!(site, "F",    F,    fermionic=false)

    return site
end
