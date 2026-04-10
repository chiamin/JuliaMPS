using Test

@testset "MPSCore" begin
    include("test_tensor_utils.jl")
    include("test_sites.jl")
    include("test_lattice.jl")
    include("test_mps.jl")
    include("test_mpo.jl")
    include("test_mps_init.jl")
    include("test_mps_operations.jl")
    include("test_compression.jl")
    include("test_auto_mpo.jl")
    include("test_environment.jl")
    include("test_effective_operators.jl")
    include("test_dmrg.jl")
    include("test_tdvp.jl")
end
