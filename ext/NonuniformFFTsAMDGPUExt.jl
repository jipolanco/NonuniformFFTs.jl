module NonuniformFFTsAMDGPUExt

using NonuniformFFTs
using NonuniformFFTs.Kernels: Kernels
using Bessels: Bessels
using AMDGPU
using AMDGPU.Device: @device_override

# Some empirical observations (on AMD MI300A, ROCm 6.3.3):
#
# - using Bessel functions defined by AMD (@device_override below) is significantly slower
#   than calling functions in Bessels.jl (at least on Float64). Moreover, GPU performance of
#   the Bessels.jl implementation can be greatly improved by avoiding branches (if/else), even
#   if this means performing more operations. This is what we do below.
#
# - direct evaluation of the BackwardsKaiserBesselKernel (sinh) is very slightly faster than the
#   KaiserBesselKernel (Bessel function I₀) and also a bit more accurate, so BKB is the default.
#
# - direct evaluation can be significantly faster than fast polynomial approximation
#   (especially for shared-memory type-2, where the difference is huge for some reason), so
#   we use Direct() evaluation by default.

# This is currently not wrapped in AMDGPU.jl, probably because besseli0 is not defined by
# SpecialFunctions.jl either (the more general besseli is defined though).
# See:
#  - https://doc-july-17.readthedocs.io/en/latest/ROCm_Compiler_SDK/ocml.html for available functions
#  - https://github.com/JuliaGPU/AMDGPU.jl/blob/master/src/device/gcn/math.jl for functions wrapped in AMDGPU.jl
# @device_override Kernels._besseli0(x::Float64) = ccall("extern __ocml_i0_f64", llvmcall, Cdouble, (Cdouble,), x)
# @device_override Kernels._besseli0(x::Float32) = ccall("extern __ocml_i0_f32", llvmcall, Cfloat, (Cfloat,), x)

# Adapted from Bessels.jl implementation; seems to be faster on AMD GPUs.
function besseli0_branchless(x::T) where {T}
    x = abs(x)
    xh = x / 2
    y_lo = let a = xh * xh
        muladd(a, evalpoly(a, Bessels.besseli0_small_coefs(T)), 1)
    end
    y_hi = let a = exp(xh)
	    xinv = inv(x)
        s = a * evalpoly(xinv, Bessels.besseli0_med_coefs(T)) * sqrt(xinv)
        a * s
    end
    ifelse(x < T(7.75), y_lo, y_hi)
end

@device_override Kernels._besseli0(x::Union{Float32, Float64}) = besseli0_branchless(x)

# This should enable minor performance gains in set_points!.
# This is not needed in the CUDA extension as CUDA.jl already overrides `rem` to call CUDA functions.
# TODO: add this to AMDGPU.jl
@device_override Base.rem(x::Float32, y::Float32) = ccall("extern __ocml_fmod_f32", llvmcall, Cfloat, (Cfloat, Cfloat), x, y)
@device_override Base.rem(x::Float64, y::Float64) = ccall("extern __ocml_fmod_f64", llvmcall, Cdouble, (Cdouble, Cdouble), x, y)

NonuniformFFTs.default_kernel(::ROCBackend) = BackwardsKaiserBesselKernel()

NonuniformFFTs.default_kernel_evalmode(::ROCBackend) = Direct()

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

# This seems to be significantly faster than the default.
NonuniformFFTs.groupsize_spreading_gpu_shmem(::ROCBackend, Np::Integer) = 256

# For shared-memory interpolation, the MI210 and MI300A prefer a very large group size.
NonuniformFFTs.groupsize_interp_gpu_shmem(::ROCBackend) = 1024

end
