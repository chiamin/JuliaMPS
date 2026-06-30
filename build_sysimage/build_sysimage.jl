# build_sysimage.jl
#
# Build the `mps_core` sysimage: a fast base image for MPSCore work (and anything
# else built on the same numeric stack, such as the MPS tutorial).
#
# It bakes in the PUBLIC libraries MPSCore depends on — TensorOperations, KrylovKit,
# LinearAlgebra, Random — plus Plots (for figures). It deliberately does NOT bake in
# MPSCore itself: MPSCore is a thin layer over those libraries, so once they are baked
# its own time-to-first-execution is well under a second, and leaving it out means you
# never have to rebuild the sysimage after editing MPSCore's source.
#
# Usage:
#     cd MPSCore/build_sysimage
#     julia --project=. build_sysimage.jl              # -> ~/.julia/sysimages/mps_core.so
#     julia --project=. build_sysimage.jl /custom/path.so
#
# The workload runs MPSCore's examples to trace the exact KrylovKit / TensorOperations
# code paths (DMRG eigsolve, TDVP exponentiate, @tensor contractions) into the image.

using PackageCompiler

const HERE     = @__DIR__
const WORKLOAD = joinpath(HERE, "precompile_workload.jl")
const DEFAULT_OUT = joinpath(homedir(), ".julia", "sysimages", "mps_core.so")
const SYSIMG_PATH = isempty(ARGS) ? DEFAULT_OUT : abspath(ARGS[1])

mkpath(dirname(SYSIMG_PATH))

# PUBLIC libraries only — NOT MPSCore (see header).
const PACKAGES = [
    :LinearAlgebra,
    :Random,
    :TensorOperations,
    :KrylovKit,
    :Plots,
]

@info "Building mps_core sysimage" SYSIMG_PATH PACKAGES WORKLOAD

create_sysimage(
    PACKAGES;
    sysimage_path             = SYSIMG_PATH,
    precompile_execution_file = WORKLOAD,
    # On low-RAM machines the final object-generation step can be OOM-killed
    # (ProcessSignaled 9). -O1 markedly lowers the compiler's peak memory use.
    sysimage_build_args       = `-O1`,
)

println()
println("Done: ", SYSIMG_PATH)
println("Point a Julia kernel / REPL at it with  -J ", SYSIMG_PATH)
