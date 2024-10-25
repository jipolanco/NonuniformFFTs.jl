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

end
