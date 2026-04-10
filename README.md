# MPSCore.jl

A dense-only MPS / MPO toolkit for DMRG and TDVP, written in Julia.

`MPSCore` is a Julia rewrite of the dense subset of [UnitenDMRG (Python +
cytnx)](https://github.com/) — it focuses on plain `Array{T,N}` tensors
contracted via [`TensorOperations.jl`](https://github.com/Jutho/TensorOperations.jl)
and [`KrylovKit.jl`](https://github.com/Jutho/KrylovKit.jl) for the local
eigensolver / time evolution.  No quantum-number block sparsity, no symmetry
sectors — just clean, easy-to-read code suitable for teaching, prototyping,
and small/medium-scale simulations.

## Features

- **Open-boundary MPS / MPO** with rank-3 / rank-4 dense tensors
- **AutoMPO** with automatic Jordan–Wigner string insertion
- **DMRG** ground state and excited states (penalty method),
  1-site and 2-site sweeps, real and complex Hamiltonians
- **TDVP** real-time and imaginary-time evolution, 1-site and 2-site
- **`fit_apply_mpo`** variational MPO–MPS application
- **MPS / MPO compression** via SVD truncation
- Built-in physical sites: spin-1/2, spinless fermion, electron (Hubbard)
- 2D `SquareLattice` helper for nearest-neighbour bonds
- 364 unit tests covering every module

## Installation

The package is not (yet) registered.  Clone and use it from a local path:

```julia
using Pkg
Pkg.develop(path="/path/to/MPSCore")
using MPSCore
```

Dependencies: `LinearAlgebra` (stdlib), `TensorOperations`, `KrylovKit`,
Julia ≥ 1.10.

## Quick start: Heisenberg DMRG

```julia
using MPSCore

N = 10
site = spin_half()

# Build the Hamiltonian via AutoMPO
ampo = AutoMPO(N, site)
for i in 1:N-1
    add!(ampo, 1.0, "Sz", i, "Sz", i+1)
    add!(ampo, 0.5, "Sp", i, "Sm", i+1)
    add!(ampo, 0.5, "Sm", i, "Sp", i+1)
end
H = to_mpo(ampo)

# Random initial state, right-canonical form
psi = random_mps(Float64, N, 2, 8)
orthogonalize!(psi)

# DMRG schedule: (max bond dim, cutoff) per sweep
energies = dmrg!(psi, H;
                 sweeps   = 5,
                 max_dims = [10, 20, 40, 40, 40],
                 cutoffs  = [0.0, 0.0, 1e-10, 1e-10, 1e-10])

println("Final energy = ", energies[end])
println("E / N        = ", energies[end] / N)
```

## Conventions

### Tensor leg orders (fixed)

```
MPS site : Array{T,3},  dims = (l, i, r)
MPO site : Array{T,4},  dims = (l, ip, i, r)
Env (op) : Array{T,3},  dims = (dn, w, up)     # ket virtual, MPO virtual, bra virtual
Env (vec): Array{T,2},  dims = (dn, up)
```

`ip` = outgoing physical index (bra side, contracted with `conj(MPS)`)
`i`  = incoming physical index (ket side)

### Bra / ket

The bra side **always** uses `conj(...)`.  This is the equivalent of cytnx's
`.Dagger()` and is enforced for both real and complex tensors.

### Indexing

Sites are 1-indexed throughout (Julia convention): `psi[1]`, `psi[N]`,
`add!(ampo, J, "Sz", 1, "Sz", 2)`, `SquareLattice` coordinates `(x, y)` with
`x ∈ 1..Lx`, `y ∈ 1..Ly`.

### Spin-1/2 basis

```
index 1 = |↑⟩  (Sz = +1/2)
index 2 = |↓⟩  (Sz = -1/2)

Sz = (1/2) σ_z = diag(+0.5, -0.5)
[Sz, S±] = ±S±    (standard physics convention)
```

## Project layout

```
src/
├── MPSCore.jl              # package entry point (module + exports)
├── Tensor/
│   └── TensorUtils.jl      # svd_split, qr_split, direct_sum
├── Sites/
│   ├── Site.jl             # PhysicalSite base
│   ├── SpinHalf.jl
│   ├── SpinlessFermion.jl
│   └── Electron.jl
├── Lattice/
│   └── Square.jl
├── MPS/
│   ├── MPS.jl              # state, gauge, make_phi / update_sites!
│   ├── MPO.jl
│   ├── Init.jl             # random_mps, product_state
│   ├── Operations.jl       # inner, expectation, mps_sum, exact_apply_mpo,
│   │                       #  mpo_product, fit_apply_mpo, ...
│   ├── Compression.jl      # svd_compress_mps
│   ├── MPOCompression.jl   # svd_compress_mpo
│   └── AutoMPO.jl          # FSM-based MPO construction
├── DMRG/
│   ├── Environment.jl      # OperatorEnv, VectorEnv (with stale window)
│   ├── EffectiveOperators.jl   # EffOperator, EffVector
│   └── Engine.jl           # DMRGEngine, sweep!, dmrg!
└── TDVP/
    └── Engine.jl           # TDVPEngine
test/
└── ... (one test file per module, 364 tests total)
examples/
└── ... (small runnable scripts — see below)
```

## Examples

The `examples/` directory contains short, self-contained scripts:

| File | What it shows |
|---|---|
| `example_product_state.jl` | Build a product-state MPS, compute its norm |
| `example_dmrg.jl`          | Heisenberg DMRG ground state, schedule + Bethe ansatz comparison |
| `example_hubbard.jl`       | 1D Hubbard MPO via AutoMPO, full ED comparison |
| `example_fermion_2d.jl`    | 2D spinless-fermion MPO on a square lattice (JW + ED) |
| `example_tdvp.jl`          | Real-time TDVP, energy/norm conservation |
| `example_mpo_product.jl`   | `mpo_product` + `svd_compress_mpo` for `H²` |

Run an example:

```bash
cd MPSCore
julia --project=. examples/example_dmrg.jl
```

## Tests

```bash
cd MPSCore
julia --project=. test/runtests.jl
```

All 364 tests should pass in roughly 3 minutes.

## Differences from the original (cytnx) UnitenDMRG

| Feature | Python (cytnx) | Julia (MPSCore) |
|---|---|---|
| Tensor backend | `cytnx.UniTensor` (labelled, QN-aware) | Plain `Array{T,N}` + `@tensor` |
| Quantum numbers / block sparsity | Yes | No (dense only) |
| Local eigensolver | Custom Davidson | `KrylovKit.eigsolve` |
| Time evolution | Custom Lanczos `expm` | `KrylovKit.exponentiate` |
| Site indexing | 0-indexed | 1-indexed |
| Density-matrix MPS compression | Implemented (blocked by cytnx bug) | Not implemented |
| `fit_mpo_product` | Implemented | Not implemented (stub) |

## License

Same as the parent project.
