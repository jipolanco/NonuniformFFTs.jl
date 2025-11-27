# Test GPU code using CPU arrays. By default everything is run on the CPU using OpenCL.jl.

# We test with OpenCL on CPU, but this requires a modified version of the OpenCL.jl
# sources to enable support for float atomics (see Project.toml). Also, note that the
# LocalPreferences.toml file sets default_memory_backend = "svm" (unified memory by
# default), which is needed for FFTs to work (using FFTW).
#
# To test an actual GPU backend, set the environment variable JULIA_GPU_BACKEND before
# launching this script. Possible values are:
#  - CUDA
#  - AMDGPU
# The required packages (e.g. CUDA.jl) will automatically be installed in the current
# environment.

using NonuniformFFTs
using ThreadPinning: ThreadPinning  # this might help pinning threads used by pocl_jll
using KernelAbstractions: KernelAbstractions as KA
using AbstractFFTs: fftfreq
using AbstractNFFTs: AbstractNFFTs
using Adapt: Adapt, adapt
using GPUArraysCore: AbstractGPUArray
using OpenCL, pocl_jll
using Random
using Test

ThreadPinning.pinthreads(:cores)

# Print OpenCL information
OpenCL.versioninfo()
@show cl.platform()
@show cl.device()
println(stdout, "Extensions:")
show(stdout, MIME"text/plain"(), cl.device().extensions)
println(stdout)

# Allow testing on actual GPU arrays if the right environment variable is passed (and the
# right package is installed).
const GPU_BACKEND = get(ENV, "JULIA_GPU_BACKEND", "none")

# ================================================================================ #

array_type(::OpenCLBackend) = CLArray

@static if GPU_BACKEND == "CUDA"
    using Pkg; Pkg.add("CUDA")
    using CUDA
    const GPUBackend = CUDABackend
    array_type(::CUDABackend) = CuArray
elseif GPU_BACKEND == "AMDGPU"
    using Pkg; Pkg.add("AMDGPU")
    using AMDGPU
    const GPUBackend = ROCBackend
    array_type(::ROCBackend) = ROCArray
else
    const GPUBackend = nothing
end

const tested_backends = Any[]

if GPUBackend !== nothing
    push!(tested_backends, GPUBackend())
    @info "GPU tests - using:" GPU_BACKEND GPUBackend
end

if VERSION ≥ v"1.12"
    push!(tested_backends, OpenCLBackend())
end

function run_plan(
        p::PlanNUFFT, xp_init::Tuple, vp_init::NTuple{Nc, AbstractVector};
        callbacks_type1 = NUFFTCallbacks(), callbacks_type2 = NUFFTCallbacks(),
    ) where {Nc}
    (; backend,) = p

    xp = adapt(backend, xp_init)
    vp = adapt(backend, vp_init)

    set_points!(p, xp)

    T = eltype(p)  # type in Fourier space (always complex) - for compatibility with AbstractNFFTs plans
    @test T <: Complex
    dims = size(p)
    us = map(_ -> KA.allocate(backend, T, dims), vp)
    exec_type1!(us, p, vp; callbacks = callbacks_type1)

    wp = map(similar, vp)
    exec_type2!(wp, p, us; callbacks = callbacks_type2)

    (; backend, us, wp,)
end

# Test that block_dims_gpu_shmem returns compile-time constants.
function test_inference_block_dims_shmem(backend, ::Type{Z}, dims, m::HalfSupport) where {Z}
    batch_size = Val(NonuniformFFTs.default_gpu_batch_size(backend))
    ret = NonuniformFFTs.block_dims_gpu_shmem(backend, Z, dims, m, batch_size)
    Val(ret)
end

function compare_with_cpu(
        ::Type{T}, dims;
        backend = GPUBackend(),
        Np = prod(dims), ntransforms::Val{Nc} = Val(1),
        callbacks_type1 = NUFFTCallbacks(),
        callbacks_type2 = NUFFTCallbacks(),
        kws...
    ) where {T <: Number, Nc}
    # Generate some non-uniform random data on the CPU
    rng = Xoshiro(42)
    D = length(dims)    # number of dimensions
    Tr = real(T)
    xp_init = ntuple(_ -> [rand(rng, Tr) * Tr(2π) for _ ∈ 1:Np], D)  # non-uniform points in [0, 2π]ᵈ
    vp_init = ntuple(_ -> randn(rng, T, Np), ntransforms)

    @inferred test_inference_block_dims_shmem(backend, T, dims, HalfSupport(4))

    params = (; m = HalfSupport(4), kernel = KaiserBesselKernel(), σ = 1.5, ntransforms, kws...)
    p_cpu = @inferred PlanNUFFT(T, dims; params..., backend = CPU())
    p_gpu = if backend isa OpenCLBackend
        PlanNUFFT(T, dims; params..., backend = backend)  # not fully inferred (the M is not inferred in CLArray{T, N, M})
    else
        @inferred PlanNUFFT(T, dims; params..., backend = backend)
    end

    # Test that plan_nfft interface works.
    @testset "AbstractNFFTs.plan_nfft" begin
        xmat = hcat(xp_init...)'  # matrix of dimensions (D, Np)
        p_nfft = if backend isa OpenCLBackend
            AbstractNFFTs.plan_nfft(array_type(backend), xmat, dims)
        else
            @inferred AbstractNFFTs.plan_nfft(array_type(backend), xmat, dims)
        end
        @test typeof(p_nfft.p.backend) === typeof(backend)
        # Test without the initial argument (type of array)
        p_nfft_cpu = @inferred AbstractNFFTs.plan_nfft(xmat, dims)
        @test p_nfft_cpu.p.backend isa CPU
    end

    r_cpu = run_plan(p_cpu, xp_init, vp_init; callbacks_type1, callbacks_type2)
    r_gpu = run_plan(
        p_gpu, xp_init, vp_init;
        callbacks_type1 = adapt(backend, callbacks_type1),
        callbacks_type2 = adapt(backend, callbacks_type2),
    )

    # The differences of the order of 1e-7 (= roughly the expected accuracy given the
    # chosen parameters) are explained by the fact that the CPU uses a polynomial
    # approximation of the KB kernel, while the GPU evaluates it "exactly" from its
    # definition (based on Bessel functions for KB).
    rtol = Tr === Float64 ? 1e-7 : Tr === Float32 ? 1f-5 : nothing

    @testset "Type 1" begin
        for c ∈ 1:Nc
            @test r_cpu.us[c] ≈ Array(r_gpu.us[c]) rtol=rtol  # output of type-1 transform
        end
    end

    @testset "Type 2" begin
        for c ∈ 1:Nc
            @test r_cpu.wp[c] ≈ Array(r_gpu.wp[c]) rtol=rtol  # output of type-2 transform
        end
    end

    nothing
end

# Defining callbacks as a callable struct is needed for testing callbacks on CPU and GPU
# with captured arrays (e.g. ks, weights). Using adapt(CUDABackend(), ::NonuniformCallbackMultiplyByWeights)
# transforms the weights array into a CUDA array.
struct NonuniformCallbackMultiplyByWeights{Weights <: AbstractArray} <: Function
    weights :: Weights
end
(f::NonuniformCallbackMultiplyByWeights)(v, n) = oftype(v, @inbounds(v .* f.weights[n]))

function test_gpu(backend::KA.Backend)
    dims = (35, 64, 40)
    @testset "T = $T" for T ∈ (Float32, ComplexF32)
        compare_with_cpu(T, dims; backend)
    end
    @testset "sort_points = $sort_points" for sort_points ∈ (False(), True())
        compare_with_cpu(Float64, dims; backend, sort_points)
    end
    @testset "No blocking" begin  # spatial sorting disabled
        compare_with_cpu(ComplexF64, dims; backend, block_size = nothing)
    end
    @testset "gpu_method = :shared_memory" begin
        @testset "Float32" compare_with_cpu(Float32, dims; backend, gpu_method = :shared_memory)
        @testset "ComplexF32" compare_with_cpu(ComplexF32, dims; backend, gpu_method = :shared_memory)
    end
    @testset "Multiple transforms" begin
        ntransforms = Val(2)
        @testset "Global memory" compare_with_cpu(Float32, dims; backend, ntransforms, gpu_method = :global_memory)
        @testset "Shared memory" compare_with_cpu(Float32, dims; backend, ntransforms, gpu_method = :shared_memory)
    end
    @testset "Callbacks" begin
        Np = prod(dims) ÷ 2
        ks = map(N -> fftfreq(N, N), dims)
        weights = rand(Xoshiro(42), Np)  # note: this needs to be moved to the GPU for GPU transforms (we use Adapt for this)
        nonuniform = NonuniformCallbackMultiplyByWeights(weights)
        uniform = let ks = ks
            @inline function (w, idx)
                k⃗ = @inbounds getindex.(ks, idx)
                k² = sum(abs2, k⃗)
                factor = ifelse(iszero(k²), zero(k²), inv(k²))  # divide by k² but avoiding division by zero
                oftype(w, w .* factor)
            end
        end
        callbacks = NUFFTCallbacks(;
            nonuniform,
            uniform,
        )
        @testset "Global memory" compare_with_cpu(ComplexF32, dims; backend, Np, callbacks_type1 = callbacks, callbacks_type2 = callbacks, gpu_method = :global_memory)
        @testset "Shared memory" compare_with_cpu(ComplexF32, dims; backend, Np, callbacks_type1 = callbacks, callbacks_type2 = callbacks, gpu_method = :shared_memory)
        @testset "Shared memory (ntransforms = 2)" compare_with_cpu(ComplexF32, dims; backend, Np, ntransforms = Val(2), callbacks_type1 = callbacks, callbacks_type2 = callbacks, gpu_method = :shared_memory)
    end
    nothing
end

@testset "GPU implementation (using $backend)" for backend in tested_backends
    test_gpu(backend)
end
