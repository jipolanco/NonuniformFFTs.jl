# Test GPU code using CPU arrays. Everything is run on the CPU.
#
# We define a minimal custom array type, so that it runs the kernels the GPU would run
# instead of using the alternative CPU branches.
# We also tried to use JLArrays instead, but we need scalar indexing which is disallowed by
# JLArray (since it's supposed to mimic GPU arrays). Even with allowscalar(true), kernels
# seem to fail for other reasons.

using NonuniformFFTs
using StaticArrays: SVector  # for convenience
using KernelAbstractions: KernelAbstractions as KA
using Adapt: Adapt, adapt
using GPUArraysCore: AbstractGPUArray
using Random
using Test

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

function run_plan(p::PlanNUFFT, xp_init::AbstractArray, vp_init::AbstractVector)
    (; backend,) = p

    xp = adapt(backend, xp_init)
    vp = adapt(backend, vp_init)

    set_points!(p, xp)

    T = eltype(p)  # this is actually defined in AbstractNFFTs; it represents the type in Fourier space (always complex)
    @test T <: Complex
    dims = size(p)
    us = KA.allocate(backend, T, dims)
    exec_type1!(us, p, vp)

    wp = similar(vp)
    exec_type2!(wp, p, us)

    (; backend, us, wp,)
end

function compare_with_cpu(::Type{T}, dims; Np = prod(dims)) where {T <: Number}
    # Generate some non-uniform random data on the CPU
    rng = Xoshiro(42)
    N = length(dims)    # number of dimensions
    Tr = real(T)
    xp_init = [rand(rng, SVector{N, Tr}) * Tr(2π) for _ ∈ 1:Np]  # non-uniform points in [0, 2π]ᵈ
    vp_init = randn(rng, T, Np)          # random values at points

    params = (; m = HalfSupport(4), kernel = KaiserBesselKernel(), σ = 1.5,)
    p_cpu = PlanNUFFT(T, dims; params..., backend = CPU())
    p_gpu = PlanNUFFT(T, dims; params..., backend = PseudoGPU())

    r_cpu = run_plan(p_cpu, xp_init, vp_init)
    r_gpu = run_plan(p_gpu, xp_init, vp_init)

    @test r_cpu.us ≈ r_gpu.us  # output of type-1 transform
    @test r_cpu.wp ≈ r_gpu.wp  # output of type-2 transform

    nothing
end

@testset "GPU implementation (using CPU)" begin
    types = (Float32, ComplexF32)
    dims = (35, 64, 40)
    @testset "T = $T" for T ∈ types
        compare_with_cpu(T, dims)
    end
end
