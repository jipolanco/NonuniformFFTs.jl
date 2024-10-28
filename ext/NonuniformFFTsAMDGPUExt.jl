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

# We want the result of this function to be a compile-time constant to avoid some type
# instabilities, which is why we hardcode the result even though it could be obtained using
# the AMDGPU API.
# On AMDGPU, the static shared memory size (called LDS = Local Data Storage) is usually 64 KiB.
# This is the case in particular of all current AMD Instinct GPUs/APUs (up to the CDNA3
# architecture at least, i.e. MI300* models).
# See https://rocm.docs.amd.com/en/latest/reference/gpu-arch-specs.html
function NonuniformFFTs.available_static_shared_memory(::ROCBackend)
    expected = Int32(64) << 10  # 64 KiB
    actual = AMDGPU.HIP.attribute(AMDGPU.device(), AMDGPU.HIP.hipDeviceAttributeMaxSharedMemoryPerBlock)
    expected == actual || @warn(lazy"AMDGPU device reports non-standard shared memory size: $actual bytes")
    expected
end

end
