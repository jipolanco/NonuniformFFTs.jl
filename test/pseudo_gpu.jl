# Test GPU code using CPU arrays. By default everything is run on the CPU.
#
# We define a minimal custom array type, so that it runs the kernels the GPU would run
# instead of using the alternative CPU branches.
# We also tried to use JLArrays instead, but we need scalar indexing which is disallowed by
# JLArray (since it's supposed to mimic GPU arrays). Even with allowscalar(true), kernels
# seem to fail for other reasons.
#
# To test an actual GPU backend, set the environment variable JULIA_GPU_BACKEND before
# launching this script. Possible values are:
#  - PseudoGPU (default)
#  - CUDA
#  - AMDGPU
# The required packages (e.g. CUDA.jl) will automatically be installed in the current
# environment.


# TODO:
# - use new JLBackend in the latest GPUArrays

using NonuniformFFTs
using StaticArrays: SVector  # for convenience
using KernelAbstractions: KernelAbstractions as KA
using AbstractNFFTs: AbstractNFFTs
using Adapt: Adapt, adapt
using GPUArraysCore: AbstractGPUArray
using Random
using Test

# Allow testing on actual GPU arrays if the right environment variable is passed (and the
# right package is installed).
const GPU_BACKEND = get(ENV, "JULIA_GPU_BACKEND", "PseudoGPU")

# ================================================================================ #

# Definition of custom "GPU" array type and custom KA backend.

struct PseudoGPUArray{T, N} <: AbstractGPUArray{T, N}
    data :: Array{T, N}  # actually on the CPU
end
Base.size(u::PseudoGPUArray) = size(u.data)
Base.@propagate_inbounds Base.getindex(u::PseudoGPUArray, i...) = @inbounds u.data[i...]
Base.@propagate_inbounds Base.setindex!(u::PseudoGPUArray, v, i...) = @inbounds u.data[i...] = v
Base.resize!(u::PseudoGPUArray, n) = resize!(u.data, n)
Base.pointer(u::PseudoGPUArray, i::Integer = 1) = pointer(u.data, i)
Base.unsafe_convert(::Type{Ptr{T}}, u::PseudoGPUArray{T}) where {T} = pointer(u)
Base.similar(u::PseudoGPUArray, ::Type{T}, dims::Dims) where {T} =
    PseudoGPUArray(similar(u.data, T, dims))

struct PseudoGPU <: KA.GPU end
KA.isgpu(::PseudoGPU) = false  # needed to be considered as a CPU backend by KA
KA.get_backend(::PseudoGPUArray) = PseudoGPU()
KA.allocate(::PseudoGPU, ::Type{T}, dims::Tuple) where {T} = PseudoGPUArray(KA.allocate(KA.CPU(), T, dims))
KA.synchronize(::PseudoGPU) = nothing
Adapt.adapt_storage(::Type{<:PseudoGPUArray}, u::Array) = PseudoGPUArray(copy(u)) # simulate host → device copy (making sure arrays are not aliased)
Adapt.adapt_storage(::PseudoGPU, u::PseudoGPUArray) = u
Adapt.adapt_storage(::PseudoGPU, u) = adapt(PseudoGPUArray, u)

# Convert kernel to standard CPU kernel (relies on KA internals...)
function (kernel::KA.Kernel{PseudoGPU, GroupSize, NDRange, Fun})(args...; kws...) where {GroupSize, NDRange, Fun}
    kernel_cpu = KA.Kernel{KA.CPU, GroupSize, NDRange, Fun}(KA.CPU(), kernel.f)
    kernel_cpu(args...; kws...)
end

# ================================================================================ #

@static if GPU_BACKEND == "PseudoGPU"
    const GPUBackend = PseudoGPU
    const GPUArrayType = PseudoGPUArray
elseif GPU_BACKEND == "CUDA"
    using Pkg; Pkg.add("CUDA")
    using CUDA
    const GPUBackend = CUDABackend
    const GPUArrayType = CuArray
elseif GPU_BACKEND == "AMDGPU"
    using Pkg; Pkg.add("AMDGPU")
    using AMDGPU
    const GPUBackend = ROCBackend
    const GPUArrayType = ROCArray
else
    error("unknown value of JULIA_GPU_BACKEND: $GPU_BACKEND")
end

@info "GPU tests - using:" GPU_BACKEND GPUBackend GPUArrayType

function run_plan(p::PlanNUFFT, xp_init::AbstractArray, vp_init::NTuple{Nc, AbstractVector}) where {Nc}
    (; backend,) = p

    xp = adapt(backend, xp_init)
    vp = adapt(backend, vp_init)

    set_points!(p, xp)

    T = eltype(p)  # type in Fourier space (always complex) - for compatibility with AbstractNFFTs plans
    @test T <: Complex
    dims = size(p)
    us = map(_ -> KA.allocate(backend, T, dims), vp)
    exec_type1!(us, p, vp)

    wp = map(similar, vp)
    exec_type2!(wp, p, us)

    (; backend, us, wp,)
end

# Test that block_dims_gpu_shmem returns compile-time constants.
function test_inference_block_dims_shmem(backend, ::Type{Z}, dims, m::HalfSupport) where {Z}
    batch_size = Val(NonuniformFFTs.default_gpu_batch_size(backend))
    ret = NonuniformFFTs.block_dims_gpu_shmem(backend, Z, dims, m, batch_size)
    Val(ret)
end

function compare_with_cpu(::Type{T}, dims; Np = prod(dims), ntransforms::Val{Nc} = Val(1), kws...) where {T <: Number, Nc}
    # Generate some non-uniform random data on the CPU
    rng = Xoshiro(42)
    N = length(dims)    # number of dimensions
    Tr = real(T)
    xp_init = [rand(rng, SVector{N, Tr}) * Tr(2π) for _ ∈ 1:Np]  # non-uniform points in [0, 2π]ᵈ
    vp_init = ntuple(_ -> randn(rng, T, Np), ntransforms)

    @inferred test_inference_block_dims_shmem(GPUBackend(), T, dims, HalfSupport(4))

    params = (; m = HalfSupport(4), kernel = KaiserBesselKernel(), σ = 1.5, ntransforms, kws...)
    p_cpu = @inferred PlanNUFFT(T, dims; params..., backend = CPU())
    p_gpu = @inferred PlanNUFFT(T, dims; params..., backend = GPUBackend())

    # Test that plan_nfft interface works.
    @testset "AbstractNFFTs.plan_nfft" begin
        xmat = reinterpret(reshape, Tr, xp_init)
        p_nfft = @inferred AbstractNFFTs.plan_nfft(GPUArrayType, xmat, dims)
        @test p_nfft.p.backend isa GPUBackend
        # Test without the initial argument (type of array)
        p_nfft_cpu = @inferred AbstractNFFTs.plan_nfft(xmat, dims)
        @test p_nfft_cpu.p.backend isa CPU
    end

    r_cpu = run_plan(p_cpu, xp_init, vp_init)
    r_gpu = run_plan(p_gpu, xp_init, vp_init)

    for c ∈ 1:Nc
        # The differences of the order of 1e-7 (= roughly the expected accuracy given the
        # chosen parameters) are explained by the fact that the CPU uses a polynomial
        # approximation of the KB kernel, while the GPU evaluates it "exactly" from its
        # definition (based on Bessel functions for KB).
        rtol = Tr === Float64 ? 1e-7 : Tr === Float32 ? 1f-5 : nothing
        @test r_cpu.us[c] ≈ Array(r_gpu.us[c]) rtol=rtol  # output of type-1 transform
        @test r_cpu.wp[c] ≈ Array(r_gpu.wp[c]) rtol=rtol  # output of type-2 transform
    end

    nothing
end

@testset "GPU implementation (using $GPU_BACKEND backend)" begin
    dims = (35, 64, 40)
    @testset "T = $T" for T ∈ (Float32, ComplexF32)
        compare_with_cpu(T, dims)
    end
    @testset "sort_points = $sort_points" for sort_points ∈ (False(), True())
        compare_with_cpu(Float64, dims; sort_points)
    end
    @testset "No blocking" begin  # spatial sorting disabled
        compare_with_cpu(ComplexF64, dims; block_size = nothing)
    end
    @testset "gpu_method = :shared_memory" begin
        @testset "Float32" compare_with_cpu(Float32, dims; gpu_method = :shared_memory)
        @testset "ComplexF32" compare_with_cpu(ComplexF32, dims; gpu_method = :shared_memory)
    end
    @testset "Multiple transforms" begin
        ntransforms = Val(2)
        @testset "Global memory" compare_with_cpu(Float32, dims; ntransforms, gpu_method = :global_memory)
        @testset "Shared memory" compare_with_cpu(Float32, dims; ntransforms, gpu_method = :shared_memory)
    end
end
