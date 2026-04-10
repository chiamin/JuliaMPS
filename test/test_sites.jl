using Test
using LinearAlgebra

include("../src/Sites/Site.jl")
include("../src/Sites/SpinHalf.jl")
include("../src/Sites/SpinlessFermion.jl")
include("../src/Sites/Electron.jl")

# ---------------------------------------------------------------------------
# PhysicalSite interface
# ---------------------------------------------------------------------------

@testset "PhysicalSite interface" begin
    site = PhysicalSite(2, "Test")
    register_op!(site, "X", [0.0 1.0; 1.0 0.0])
    register_op!(site, "F", [1.0 0.0; 0.0 -1.0], fermionic=true)

    @test site.dim == 2
    @test site.type_name == "Test"
    @test op(site, "X") ≈ [0 1; 1 0]
    @test eltype(op(site, "X")) == ComplexF64
    @test is_fermionic(site, "X") == false
    @test is_fermionic(site, "F") == true
    @test is_fermionic(site, "missing") == false   # default false for unknown

    # op() throws on missing name
    @test_throws ErrorException op(site, "missing")

    # Wrong size throws
    @test_throws ErrorException register_op!(site, "Bad", [1.0 0.0 0.0; 0.0 1.0 0.0])
end

# ---------------------------------------------------------------------------
# SpinHalf
# ---------------------------------------------------------------------------

@testset "SpinHalf" begin
    s = spin_half()

    @test s.dim == 2
    @test s.type_name == "SpinHalf"

    I2 = op(s, "I")
    Sz = op(s, "Sz")
    Sp = op(s, "Sp")
    Sm = op(s, "Sm")

    # Algebra: Sp*Sm + Sm*Sp = I
    @test isapprox(Sp*Sm + Sm*Sp, I2; atol=1e-14)

    # Standard commutators: [Sz, Sp] = +Sp, [Sz, Sm] = -Sm
    @test isapprox(Sz*Sp - Sp*Sz, Sp; atol=1e-14)
    @test isapprox(Sz*Sm - Sm*Sz, -Sm; atol=1e-14)

    # Specific matrix elements
    # Basis: index 1=|up⟩ (Sz=+1/2), index 2=|dn⟩ (Sz=-1/2)
    # Sp|dn>=|up> means Sp[1,2]=1 (row=up=1, col=dn=2)
    @test isapprox(Sp[1,2], 1.0; atol=1e-14)
    @test isapprox(Sp[1,1], 0.0; atol=1e-14)
    @test isapprox(Sp[2,1], 0.0; atol=1e-14)
    @test isapprox(Sp[2,2], 0.0; atol=1e-14)

    # Sm|up>=|dn> means Sm[2,1]=1 (row=dn=2, col=up=1)
    @test isapprox(Sm[2,1], 1.0; atol=1e-14)

    # Sz diagonal: up=+1/2 (index 1), dn=-1/2 (index 2)
    @test isapprox(Sz[1,1], 0.5; atol=1e-14)
    @test isapprox(Sz[2,2], -0.5; atol=1e-14)

    # All non-fermionic
    @test is_fermionic(s, "I")  == false
    @test is_fermionic(s, "Sz") == false
    @test is_fermionic(s, "Sp") == false
    @test is_fermionic(s, "Sm") == false
end

# ---------------------------------------------------------------------------
# SpinlessFermion
# ---------------------------------------------------------------------------

@testset "SpinlessFermion" begin
    s = spinless_fermion()

    @test s.dim == 2
    @test s.type_name == "SpinlessFermion"

    I2   = op(s, "I")
    N    = op(s, "N")
    C    = op(s, "C")
    Cdag = op(s, "Cdag")
    F    = op(s, "F")

    # {C, Cdag} = I (anticommutator)
    @test isapprox(C*Cdag + Cdag*C, I2; atol=1e-14)

    # N = Cdag * C
    @test isapprox(Cdag*C, N; atol=1e-14)

    # F = I - 2*N = (-1)^N
    @test isapprox(I2 - 2*N, F; atol=1e-14)

    # F^2 = I
    @test isapprox(F*F, I2; atol=1e-14)

    # Specific matrix elements
    # Basis: 1=|0⟩ (empty), 2=|1⟩ (occupied)
    # C|1>=|0>: C[1,2]=1
    @test isapprox(C[1,2], 1.0; atol=1e-14)
    @test isapprox(C[2,1], 0.0; atol=1e-14)
    # Cdag|0>=|1>: Cdag[2,1]=1
    @test isapprox(Cdag[2,1], 1.0; atol=1e-14)

    # N eigenvalues: |0>=0, |1>=1
    @test isapprox(N[1,1], 0.0; atol=1e-14)
    @test isapprox(N[2,2], 1.0; atol=1e-14)

    # Fermionic flags
    @test is_fermionic(s, "I")    == false
    @test is_fermionic(s, "N")    == false
    @test is_fermionic(s, "C")    == true
    @test is_fermionic(s, "Cdag") == true
    @test is_fermionic(s, "F")    == false
end

# ---------------------------------------------------------------------------
# Electron
# ---------------------------------------------------------------------------

@testset "Electron" begin
    s = electron()

    @test s.dim == 4
    @test s.type_name == "Electron"

    I4     = op(s, "I")
    Cup    = op(s, "Cup")
    Cupdag = op(s, "Cupdag")
    Cdn    = op(s, "Cdn")
    Cdndag = op(s, "Cdndag")
    Nup    = op(s, "Nup")
    Ndn    = op(s, "Ndn")
    Ntot   = op(s, "Ntot")
    Sz     = op(s, "Sz")
    F      = op(s, "F")

    # {Cup, Cupdag} = I
    @test isapprox(Cup*Cupdag + Cupdag*Cup, I4; atol=1e-14)
    # {Cdn, Cdndag} = I
    @test isapprox(Cdn*Cdndag + Cdndag*Cdn, I4; atol=1e-14)
    # {Cup, Cdn} = 0 (mixed anticommutator)
    @test isapprox(Cup*Cdn + Cdn*Cup, zeros(4,4); atol=1e-14)

    # Number operators
    @test isapprox(Cupdag*Cup, Nup; atol=1e-14)
    @test isapprox(Cdndag*Cdn, Ndn; atol=1e-14)
    @test isapprox(Nup + Ndn, Ntot; atol=1e-14)
    @test isapprox((Nup - Ndn)/2, Sz; atol=1e-14)

    # F = (-1)^Ntot
    @test isapprox(F*F, I4; atol=1e-14)
    # Diagonal of F: |0>=+1, |↑>=-1, |↓>=-1, |↑↓>=+1
    @test isapprox(diag(real.(F)), [1.0, -1.0, -1.0, 1.0]; atol=1e-14)

    # Specific matrix elements (basis: 1=|0⟩, 2=|↑⟩, 3=|↓⟩, 4=|↑↓⟩)
    # Cup: |0⟩←|↑⟩ and |↓⟩←|↑↓⟩
    @test isapprox(Cup[1,2], 1.0; atol=1e-14)
    @test isapprox(Cup[3,4], 1.0; atol=1e-14)
    # Cdn: |0⟩←|↓⟩ and |↑⟩←|↑↓⟩ with sign -1
    @test isapprox(Cdn[1,3], 1.0; atol=1e-14)
    @test isapprox(Cdn[2,4], -1.0; atol=1e-14)

    # Fermionic flags
    @test is_fermionic(s, "I")      == false
    @test is_fermionic(s, "Cup")    == true
    @test is_fermionic(s, "Cupdag") == true
    @test is_fermionic(s, "Cdn")    == true
    @test is_fermionic(s, "Cdndag") == true
    @test is_fermionic(s, "Nup")    == false
    @test is_fermionic(s, "Ndn")    == false
    @test is_fermionic(s, "Ntot")   == false
    @test is_fermionic(s, "Sz")     == false
    @test is_fermionic(s, "F")      == false
end
