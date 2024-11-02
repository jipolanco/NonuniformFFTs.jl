module NonuniformFFTsAMDGPUExt

using NonuniformFFTs
using NonuniformFFTs.Kernels: Kernels
using KernelAbstractions: KernelAbstractions as KA
using AMDGPU
using AMDGPU.Device: @device_override

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

function NonuniformFFTs.launch_shmem_kernel(
        obj::KA.Kernel{<:ROCBackend}, args::Vararg{Any, N};
        ngroups::NTuple{D},
    ) where {N, D}
    # This is adapted from https://github.com/JuliaGPU/AMDGPU.jl/blob/master/src/ROCKernels.jl
    config = let groupsize = 256  # this is a sort of initial guess (value is not very important)
        workgroupsize = ntuple(d -> d == 1 ? groupsize : one(groupsize), Val(D))
        ndrange = workgroupsize .* ngroups
        ndrange, _, iterspace, _ = KA.launch_config(obj, ndrange, workgroupsize)
        ctx = KA.mkcontext(obj, ndrange, iterspace)
        kernel = AMDGPU.@roc launch=false obj.f(ctx, args...)
        AMDGPU.launch_configuration(kernel)
    end
    let groupsize = config.groupsize
        # Similar to above but with the updated groupsize
        workgroupsize = ntuple(d -> d == 1 ? groupsize : one(groupsize), Val(D))
        ndrange = workgroupsize .* ngroups
        obj(args...; workgroupsize, ndrange)
    end
    nothing
end

end
