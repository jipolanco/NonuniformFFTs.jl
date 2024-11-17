module NonuniformFFTsCUDAExt

using NonuniformFFTs
using NonuniformFFTs.Kernels: Kernels
using CUDA
using CUDA.CUFFT: CUFFT
using CUDA: @device_override

# This is currently not wrapped in CUDA.jl, probably because besseli0 is not defined by
# SpecialFunctions.jl either (the more general besseli is defined though).
# See:
#  - https://docs.nvidia.com/cuda/cuda-math-api/index.html for available functions
#  - https://github.com/JuliaGPU/CUDA.jl/blob/master/ext/SpecialFunctionsExt.jl for functions wrapped in CUDA.jl
@device_override Kernels._besseli0(x::Float64) = ccall("extern __nv_cyl_bessel_i0", llvmcall, Cdouble, (Cdouble,), x)
@device_override Kernels._besseli0(x::Float32) = ccall("extern __nv_cyl_bessel_i0f", llvmcall, Cfloat, (Cfloat,), x)

# Set KaiserBesselKernel as default backend on CUDA.
# It's slightly faster than BackwardsKaiserBesselKernel when using Bessel functions from CUDA (wrapped above).
# The difference is not huge though.
NonuniformFFTs.default_kernel(::CUDABackend) = KaiserBesselKernel()

# On CUDA (A100 at least), direct evaluation is faster than the "fast" approximation using
# piecewise polynomials.
NonuniformFFTs.default_kernel_evalmode(::CUDABackend) = Direct()

# We want the result of this function to be a compile-time constant to avoid some type
# instabilities, which is why we hardcode the result even though it could be obtained using
# the CUDA API.
function NonuniformFFTs.available_static_shared_memory(::CUDABackend)
    expected = Int32(48) << 10  # 48 KiB
    actual = CUDA.attribute(device(), CUDA.DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK)  # generally returns 48KiB
    expected == actual || @warn(lazy"CUDA device reports non-standard shared memory size: $actual bytes")
    expected
end

# Try to have between 64 and 256 threads, such that the number of threads is ideally larger
# than the batch size.
# TODO: maybe the value of 128 used in AMDGPU works well here as well?
function NonuniformFFTs.groupsize_spreading_gpu_shmem(::CUDABackend, Np::Integer)
    groupsize = 64
    c = min(Np, 256)
    while groupsize < c
        groupsize += 32
    end
    groupsize
end

NonuniformFFTs.groupsize_interp_gpu_shmem(::CUDABackend) = 64

# Override usual `mul!` to avoid GPU allocations.
# See https://github.com/JuliaGPU/CUDA.jl/issues/2249
# This is adapted from https://github.com/JuliaGPU/CUDA.jl/blob/a1db081cbc3d20fa3cb28a9f419b485db03a250f/lib/cufft/fft.jl#L308-L317
# but without the copy.
function NonuniformFFTs._fft_c2r!(
        y::DenseCuArray{T}, p, x::DenseCuArray{Complex{T}},
    ) where {T}
    # Perform plan (this may modify not only y, but also the input x)
    CUFFT.assert_applicable(p, x, y)
    CUFFT.unsafe_execute_trailing!(p, x, y)
    y
end

end
