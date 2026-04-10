# Init.jl
#
# Helpers for constructing MPS objects.

"""
    random_mps(T, num_sites, phys_dim, bond_dim; normalize=true, seed=nothing) -> MPS{T}

Create a random open-boundary MPS with uniform physical and bond dimensions.

- `T`         : element type, e.g. `Float64` or `ComplexF64`.
- `normalize` : if true (default), orthogonalize and normalize the MPS.
- `seed`      : optional integer seed for reproducibility.
"""
function random_mps(::Type{T}, num_sites::Int, phys_dim::Int, bond_dim::Int;
                    normalize::Bool=true,
                    seed::Union{Int,Nothing}=nothing) where {T<:Number}
    num_sites > 0 || error("num_sites must be positive.")
    phys_dim  > 0 || error("phys_dim must be positive.")
    bond_dim  > 0 || error("bond_dim must be positive.")

    seed !== nothing && Random.seed!(seed)
    bd = num_sites == 1 ? [1, 1] : [1; fill(bond_dim, num_sites - 1); 1]
    tensors = [randn(T, bd[k], phys_dim, bd[k+1]) for k in 1:num_sites]
    psi = MPS(tensors)
    if normalize
        orthogonalize!(psi)
        LinearAlgebra.normalize!(psi)
    end
    return psi
end

"""
    product_state(T, local_states::Vector{<:AbstractVector}) -> MPS{T}

Build a product-state MPS from per-site local-state vectors.
Each `local_states[k]` has length `phys_dim` (may differ between sites).
The resulting tensors have shape `(1, phys_dim, 1)`.
"""
function product_state(::Type{T}, local_states::Vector{<:AbstractVector}) where {T<:Number}
    isempty(local_states) && error("local_states must be non-empty.")
    tensors = Vector{Array{T,3}}(undef, length(local_states))
    for (k, v) in enumerate(local_states)
        d = length(v)
        A = zeros(T, 1, d, 1)
        @views A[1, :, 1] .= T.(v)
        tensors[k] = A
    end
    return MPS(tensors)
end
