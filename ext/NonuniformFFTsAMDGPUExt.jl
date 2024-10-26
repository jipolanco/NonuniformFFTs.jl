module NonuniformFFTsAMDGPUExt

using NonuniformFFTs
using NonuniformFFTs.Kernels: Kernels
using AMDGPU
using AMDGPU.Device: @device_override

# This is currently not wrapped in AMDGPU.jl, probably because besseli0 is not defined by
# SpecialFunctions.jl either (the more general besseli is defined though).
# See:
#  - https://doc-july-17.readthedocs.io/en/latest/ROCm_Compiler_SDK/ocml.html for available functions
#  - https://github.com/JuliaGPU/AMDGPU.jl/blob/master/src/device/gcn/math.jl for functions wrapped in AMDGPU.jl
# TODO: not sure this is faster than the "fallback" implementation in Bessels.jl
@device_override Kernels._besseli0(x::Float64) = ccall("extern __ocml_i0_f64", llvmcall, Cdouble, (Cdouble,), x)
@device_override Kernels._besseli0(x::Float32) = ccall("extern __ocml_i0_f32", llvmcall, Cfloat, (Cfloat,), x)

# As with CUDA, on AMDGPU the direct evaluation of the KaiserBesselKernel seems to be faster
# than the BackwardsKaiserBesselKernel.
NonuniformFFTs.default_kernel(::ROCBackend) = KaiserBesselKernel()

NonuniformFFTs.available_static_shared_memory(::ROCBackend) =
    AMDGPU.HIP.attribute(AMDGPU.device(), AMDGPU.HIP.hipDeviceAttributeMaxSharedMemoryPerBlock)

end
