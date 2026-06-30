# precompile_workload.jl
#
# Exercises the public-library code paths MPSCore (and the MPS tutorial) use most, so
# PackageCompiler traces and bakes them into the sysimage. We `using MPSCore` and run its
# examples to drive the exact KrylovKit (eigsolve / exponentiate) and TensorOperations
# (@tensor) specializations through DMRG and TDVP — even though MPSCore itself is NOT baked
# into the image, exercising it here compiles those underlying library paths.
#
# It is just a script: any error aborts the build, which is what we want.

using LinearAlgebra
using TensorOperations
using KrylovKit
using Plots
using MPSCore
using Random

Random.seed!(0)

# ---------------------------------------------------------------------------
# 1. Plain tensor / linear-algebra operations (used all over the tutorial + MPSCore)
# ---------------------------------------------------------------------------
let
    A = rand(2, 2); B = rand(2, 2)
    @tensor C[i, k] := A[i, j] * B[j, k]
    A3 = rand(2, 2, 2); B3 = rand(2, 2, 2); D3 = rand(2, 2, 2)
    @tensor T[i, k, n] := A3[i, j, m] * B3[j, k, q] * D3[m, q, n]
    reshape(A3, 4, 2); permutedims(A3, (2, 1, 3))

    M = rand(6, 4)
    svd(M); qr(M)
    H = M * M'; eigen(Hermitian(H))

    Mc = rand(ComplexF64, 4, 4)
    svd(Mc); qr(Mc); eigen(Hermitian(Mc + Mc'))
end

# ---------------------------------------------------------------------------
# 2. A Plots figure (the tutorial draws truncation-error curves)
# ---------------------------------------------------------------------------
let
    p = plot(1:10, rand(10); yscale = :log10, label = "demo", marker = :circle)
    plot!(p, 1:10, rand(10); ls = :dash, label = "demo2")
end

# ---------------------------------------------------------------------------
# 3. MPSCore building blocks (drives svd_split / canonical / compression paths)
# ---------------------------------------------------------------------------
let
    A = rand(2, 4, 6)
    svd_split(A, 2; absorb="right")
    svd_split(A, 2; absorb="left")
    svd_split(A, 2; absorb=nothing, maxdim=3, cutoff=1e-10)
    qr_split(A, 2)

    for T in (Float64, ComplexF64)
        psi = random_mps(T, 6, 2, 4)
        orthogonalize!(psi)
        normalize!(psi)
        move_center!(psi, 3)
        bond_dims(psi); phys_dims(psi); max_dim(psi)
        inner(psi, psi)
        svd_compress_mps(psi; max_dim=3, cutoff=1e-10)
    end

    site = spin_half()
    up = [1.0, 0.0]; dn = [0.0, 1.0]
    product_state(Float64, [iseven(k) ? dn : up for k in 1:6])
end

# ---------------------------------------------------------------------------
# 4. Run the shipped MPSCore examples (AutoMPO, DMRG, TDVP, Hubbard, 2D fermions,
#    MPO product / compression) to trace KrylovKit eigsolve / exponentiate end to end.
# ---------------------------------------------------------------------------
const EXAMPLES_DIR = abspath(joinpath(@__DIR__, "..", "examples"))

if isdir(EXAMPLES_DIR)
    for f in ("example_product_state.jl",
              "example_dmrg.jl",
              "example_hubbard.jl",
              "example_fermion_2d.jl",
              "example_tdvp.jl",
              "example_mpo_product.jl")
        path = joinpath(EXAMPLES_DIR, f)
        if isfile(path)
            @info "precompile workload: running $f"
            try
                Base.include(Module(), path)
            catch err
                @warn "example failed during precompile (continuing)" file=f err=err
            end
        end
    end
else
    @warn "MPSCore examples dir not found; skipping example workload" EXAMPLES_DIR
end

@info "precompile workload finished"
