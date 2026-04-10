using Test
using LinearAlgebra
using TensorOperations

include("../src/Tensor/TensorUtils.jl")
include("../src/MPS/MPS.jl")
include("../src/MPS/MPO.jl")
include("../src/MPS/Init.jl")
include("../src/MPS/Operations.jl")
include("../src/DMRG/Environment.jl")
include("../src/DMRG/EffectiveOperators.jl")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Build a trivial random MPO with the same physical dim at every site.
function make_random_mpo(::Type{T}, N::Int, d::Int, D::Int) where {T<:Number}
    md = [1; fill(D, N-1); 1]
    tensors = [randn(T, md[k], d, d, md[k+1]) for k in 1:N]
    return MPO(tensors)
end

# Build identity MPO of size N, phys d.
function make_identity_mpo(::Type{T}, N::Int, d::Int) where {T<:Number}
    Id = Matrix{T}(I, d, d)
    tensors = [reshape(Id, 1, d, d, 1) for _ in 1:N]
    return MPO(tensors)
end

# ---------------------------------------------------------------------------
# inner
# ---------------------------------------------------------------------------

@testset "inner" begin
    for T in (Float64, ComplexF64)
        psi = random_mps(T, 5, 2, 3)
        # <psi|psi> = norm^2
        @test isapprox(real(inner(psi, psi)), norm(psi)^2; atol=1e-10)

        # ComplexF64: <psi|psi> is real (modulo float noise)
        if T == ComplexF64
            @test abs(imag(inner(psi, psi))) < 1e-10
        end

        phi = random_mps(T, 5, 2, 3)
        # <psi|phi> matches manual contraction
        env = ones(T, 1, 1)
        for k in 1:5
            K = phi[k]; B = psi[k]
            @tensor env_new[dn, up] := env[x, y] * K[x, i, dn] * conj(B)[y, i, up]
            env = env_new
        end
        @test isapprox(inner(psi, phi), only(env); atol=1e-10)

        # Length mismatch
        @test_throws ErrorException inner(psi, random_mps(T, 4, 2, 3))
    end
end

# ---------------------------------------------------------------------------
# expectation
# ---------------------------------------------------------------------------

@testset "expectation with identity MPO" begin
    for T in (Float64, ComplexF64)
        psi = random_mps(T, 5, 2, 3)
        IdMPO = make_identity_mpo(T, 5, 2)
        # <psi|I|psi> = <psi|psi>
        @test isapprox(expectation(psi, IdMPO, psi), inner(psi, psi); atol=1e-10)
    end
end

@testset "expectation with random MPO" begin
    for T in (Float64, ComplexF64)
        psi = random_mps(T, 4, 2, 3)
        phi = random_mps(T, 4, 2, 3)
        W   = make_random_mpo(T, 4, 2, 2)

        # <psi|W|phi> = <psi|(W|phi>) = inner(psi, exact_apply_mpo(W, phi))
        Wphi = exact_apply_mpo(W, phi)
        @test isapprox(expectation(psi, W, phi), inner(psi, Wphi); atol=1e-10)
    end
end

# ---------------------------------------------------------------------------
# mps_sum
# ---------------------------------------------------------------------------

@testset "mps_sum" begin
    for T in (Float64, ComplexF64)
        psi = random_mps(T, 5, 2, 3; normalize=false)
        phi = random_mps(T, 5, 2, 3; normalize=false)
        s   = mps_sum(psi, phi)

        @test length(s) == 5
        # bond dims should be sum of components (interior only)
        bd = bond_dims(s)
        @test bd[1] == 1
        @test bd[end] == 1

        # <chi|sum> = <chi|psi> + <chi|phi> for any chi
        chi = random_mps(T, 5, 2, 2; normalize=false)
        lhs = inner(chi, s)
        rhs = inner(chi, psi) + inner(chi, phi)
        @test isapprox(lhs, rhs; atol=1e-10)

        # <sum|sum> = <psi|psi> + <psi|phi> + <phi|psi> + <phi|phi>
        lhs2 = inner(s, s)
        rhs2 = inner(psi, psi) + inner(psi, phi) + inner(phi, psi) + inner(phi, phi)
        @test isapprox(lhs2, rhs2; atol=1e-10)
    end
end

# ---------------------------------------------------------------------------
# mpo_sum
# ---------------------------------------------------------------------------

@testset "mpo_sum" begin
    for T in (Float64, ComplexF64)
        W1 = make_random_mpo(T, 4, 2, 2)
        W2 = make_random_mpo(T, 4, 2, 2)
        Wsum = mpo_sum(W1, W2)

        psi = random_mps(T, 4, 2, 3; normalize=false)
        phi = random_mps(T, 4, 2, 3; normalize=false)

        lhs = expectation(psi, Wsum, phi)
        rhs = expectation(psi, W1, phi) + expectation(psi, W2, phi)
        @test isapprox(lhs, rhs; atol=1e-10)
    end
end

# ---------------------------------------------------------------------------
# exact_apply_mpo
# ---------------------------------------------------------------------------

@testset "exact_apply_mpo" begin
    for T in (Float64, ComplexF64)
        psi = random_mps(T, 5, 2, 3; normalize=false)

        # Identity MPO is no-op
        Id = make_identity_mpo(T, 5, 2)
        psi_id = exact_apply_mpo(Id, psi)
        @test isapprox(inner(psi, psi_id), inner(psi, psi); atol=1e-10)
        @test isapprox(inner(psi_id, psi_id), inner(psi, psi); atol=1e-10)

        # Random MPO: <chi|exact_apply_mpo(W,psi)> == <chi|W|psi>
        W   = make_random_mpo(T, 5, 2, 2)
        chi = random_mps(T, 5, 2, 2; normalize=false)
        Wpsi = exact_apply_mpo(W, psi)
        @test isapprox(inner(chi, Wpsi), expectation(chi, W, psi); atol=1e-10)
    end
end

# ---------------------------------------------------------------------------
# mpo_product
# ---------------------------------------------------------------------------

@testset "mpo_product" begin
    for T in (Float64, ComplexF64)
        W1 = make_random_mpo(T, 4, 2, 2)
        W2 = make_random_mpo(T, 4, 2, 2)
        W12 = mpo_product(W1, W2)

        psi = random_mps(T, 4, 2, 3; normalize=false)
        phi = random_mps(T, 4, 2, 3; normalize=false)

        # <psi|W12|phi> = <psi|W1|(W2|phi>)
        W2phi = exact_apply_mpo(W2, phi)
        lhs = expectation(psi, W12, phi)
        rhs = expectation(psi, W1, W2phi)
        @test isapprox(lhs, rhs; atol=1e-10)
    end
end

# ---------------------------------------------------------------------------
# fit_apply_mpo (filled in after DMRG Environment is implemented)
# ---------------------------------------------------------------------------

@testset "fit_apply_mpo matches exact_apply_mpo" begin
    for T in (Float64, ComplexF64)
        N = 5
        psi = random_mps(T, N, 2, 3)
        W   = make_random_mpo(T, N, 2, 2)

        exact = exact_apply_mpo(W, psi)
        # Use a fitmps with enough bond dim to represent the exact result
        fit = random_mps(T, N, 2, max(3*2, 8))
        orthogonalize!(fit)

        fit_apply_mpo(W, psi, fit; num_center=2, nsweep=4,
                      max_dim=20, cutoff=1e-12)

        # <chi|fit> should equal <chi|exact> for any random chi
        chi = random_mps(T, N, 2, 4; normalize=false)
        @test isapprox(inner(chi, fit), inner(chi, exact); rtol=1e-6)
    end
end
