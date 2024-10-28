module NonuniformFFTsCUDAExt

using NonuniformFFTs
using NonuniformFFTs.Kernels: Kernels
using CUDA
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

# We want the result of this function to be a compile-time constant to avoid some type
# instabilities, which is why we hardcode the result even though it could be obtained using
# the CUDA API.
function NonuniformFFTs.available_static_shared_memory(::CUDABackend)
    expected = Int32(48) << 10  # 48 KiB
    actual = CUDA.attribute(device(), CUDA.DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK)  # generally returns 48KiB
    expected == actual || @warn(lazy"CUDA device reports non-standard shared memory size: $actual bytes")
    expected
end

end
