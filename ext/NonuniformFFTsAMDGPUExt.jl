module NonuniformFFTsAMDGPUExt

using NonuniformFFTs
using NonuniformFFTs.Kernels: Kernels
using AbstractFFTs: AbstractFFTs
using AMDGPU
using AMDGPU.rocFFT: rocFFT
using AMDGPU.Device: @device_override

# We add a type assertion to workaround inference issues in AMDGPU.jl.
# The issue was introduced in https://github.com/JuliaGPU/AMDGPU.jl/pull/728 (included since AMDGPU.jl v1.2.3).
# The type assertion will fail on AMDGPU.jl < v1.2.3 since the `B` parameter didn't exist before that.
function NonuniformFFTs.make_plan_rfft(u::ROCArray{T, N, Mem}, dims; kwargs...) where {T, N, Mem}
    p = AbstractFFTs.plan_rfft(u, dims; kwargs...)
    B = ROCArray{Complex{T}, N, Mem}
    p::rocFFT.rROCFFTPlan{T, true, false, N, N, B}
end

# Some empirical observations (on AMD MI210, ROCm 6.0.2):
#
# - using Bessel functions defined by AMD (@device_override below) is significantly slower
#   than calling functions in Bessels.jl (at least on Float64), so we disable the overrides
#   for now;
#
# - direct evaluation of the BackwardsKaiserBesselKernel (sinh) is faster than the
#   KaiserBesselKernel (Bessel function I₀), so BKB is the default;
#
# - in both cases, direct evaluation is significantly slower than fast polynomial
#   approximation (which is not the case on CUDA / A100), so we use FastApproximation by
#   default.

# This is currently not wrapped in AMDGPU.jl, probably because besseli0 is not defined by
# SpecialFunctions.jl either (the more general besseli is defined though).
# See:
#  - https://doc-july-17.readthedocs.io/en/latest/ROCm_Compiler_SDK/ocml.html for available functions
#  - https://github.com/JuliaGPU/AMDGPU.jl/blob/master/src/device/gcn/math.jl for functions wrapped in AMDGPU.jl
# @device_override Kernels._besseli0(x::Float64) = ccall("extern __ocml_i0_f64", llvmcall, Cdouble, (Cdouble,), x)
# @device_override Kernels._besseli0(x::Float32) = ccall("extern __ocml_i0_f32", llvmcall, Cfloat, (Cfloat,), x)

# This should enable minor performance gains in set_points!.
# This is not needed in the CUDA extension as CUDA.jl already overrides `rem` to call CUDA functions.
# TODO: add this to AMDGPU.jl
@device_override Base.rem(x::Float32, y::Float32) = ccall("extern __ocml_fmod_f32", llvmcall, Cfloat, (Cfloat, Cfloat), x, y)
@device_override Base.rem(x::Float64, y::Float64) = ccall("extern __ocml_fmod_f64", llvmcall, Cdouble, (Cdouble, Cdouble), x, y)

NonuniformFFTs.default_kernel(::ROCBackend) = BackwardsKaiserBesselKernel()

NonuniformFFTs.default_kernel_evalmode(::ROCBackend) = FastApproximation()

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

# This seems to be significantly faster than the default in some tests (but should be further tuned...).
NonuniformFFTs.groupsize_spreading_gpu_shmem(::ROCBackend, Np::Integer) = 256

# For shared-memory interpolation, the MI210 prefers a very large group size.
NonuniformFFTs.groupsize_interp_gpu_shmem(::ROCBackend) = 1024

end
