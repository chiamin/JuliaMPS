# SpinHalf.jl
#
# Factory function for spin-1/2 physical site.
#
# Basis (fixed):
#   index 1 = |up⟩  (Sz = +1/2)
#   index 2 = |dn⟩  (Sz = -1/2)
#
# Sz = (1/2)*sigma_z = (1/2)*[[1,0],[0,-1]]:
#   Sz|up⟩ = +1/2,  Sz|dn⟩ = -1/2

"""
    spin_half() -> PhysicalSite

Create a spin-1/2 PhysicalSite with operators I, Sz, Sp, Sm registered.

Basis:
    index 1 = |up⟩ (Sz = +1/2),  index 2 = |dn⟩ (Sz = -1/2)

Operators (row=ip/bra, col=i/ket):
    I  = identity
    Sz = (1/2)*sigma_z = diag(+1/2, -1/2)
    Sp = raising: Sp|dn⟩=|up⟩ → Sp[1,2]=1
    Sm = lowering: Sm|up⟩=|dn⟩ → Sm[2,1]=1

All operators are non-fermionic.
"""
function spin_half() :: PhysicalSite
    site = PhysicalSite(2, "SpinHalf")

    I2 = [1.0 0.0; 0.0 1.0]
    Sz = [0.5 0.0; 0.0 -0.5]          # (1/2)*sigma_z: up=+1/2 (index 1), dn=-1/2 (index 2)
    Sp = [0.0 1.0; 0.0 0.0]           # Sp|dn>=|up>: row=up(1), col=dn(2)
    Sm = [0.0 0.0; 1.0 0.0]           # Sm|up>=|dn>: row=dn(2), col=up(1)

    register_op!(site, "I",  I2, fermionic=false)
    register_op!(site, "Sz", Sz, fermionic=false)
    register_op!(site, "Sp", Sp, fermionic=false)
    register_op!(site, "Sm", Sm, fermionic=false)

    return site
end
