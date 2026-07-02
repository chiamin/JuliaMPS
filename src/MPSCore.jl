module MPSCore

using LinearAlgebra
using Random
using TensorOperations
using KrylovKit

# Phase 1: Tensor utilities
include("Tensor/TensorUtils.jl")

# Phase 2: Sites + Lattice
include("Sites/Site.jl")
include("Sites/SpinHalf.jl")
include("Sites/SpinlessFermion.jl")
include("Sites/Electron.jl")
include("Lattice/Square.jl")

# Phase 3: MPS / MPO core
include("MPS/MPS.jl")
include("MPS/MPO.jl")
include("MPS/Init.jl")
# Operations.jl uses OperatorEnv / EffOperator inside fit_apply_mpo,
# so it must come AFTER DMRG/Environment.jl and DMRG/EffectiveOperators.jl.
# It is included below in the DMRG section.
include("MPS/Compression.jl")
include("MPS/MPOCompression.jl")

# Phase 4: AutoMPO
include("MPS/AutoMPO.jl")

# Phase 5: DMRG
include("DMRG/Environment.jl")
include("DMRG/EffectiveOperators.jl")
include("MPS/Operations.jl")        # depends on Environment + EffOperator
include("DMRG/Engine.jl")
include("DMRG/SubspaceExpansion.jl")

# Phase 6: TDVP
include("TDVP/Engine.jl")

# ---------------------------------------------------------------------------
# Public exports
# ---------------------------------------------------------------------------

# TensorUtils
export svd_split, qr_split, direct_sum

# Sites
export AbstractSite, PhysicalSite, register_op!, op, is_fermionic
export spin_half, spinless_fermion, electron

# Lattice
export SquareLattice, site_index, site_coord, bonds

# MPS / MPO
export MPS, MPO
export phys_dims, bond_dims, mpo_dims, max_dim, center
export set_sites!, register_callback!, check_mps_tensor
export check_left_right_orthonormal
export orthogonalize!, move_center!, make_phi, update_sites!

# Init
export random_mps, product_state

# Operations
export inner, expectation, mps_sum, mpo_sum
export exact_apply_mpo, mpo_product, fit_apply_mpo

# Compression
export svd_compress_mps, svd_compress_mpo

# AutoMPO
export AutoMPO, add!, to_mpo

# DMRG
export OperatorEnv, VectorEnv, update_envs!, getenv
export EffOperator, EffVector, apply, add_term!
export DMRGEngine, sweep!, dmrg!
export expansion_term, expand_bond

# TDVP
export TDVPEngine

end # module MPSCore
