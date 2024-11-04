module NonuniformFFTs

using AbstractFFTs: AbstractFFTs
using AbstractNFFTs: AbstractNFFTs, AbstractNFFTPlan
using GPUArraysCore: AbstractGPUArray, AbstractGPUVector
using StructArrays: StructArrays, StructVector
using AbstractFFTs: AbstractFFTs
using KernelAbstractions: KernelAbstractions as KA, CPU, GPU, @kernel, @index, @Const,
    @groupsize, @localmem, @synchronize, @uniform
using Atomix: Atomix
using FFTW: FFTW
using LinearAlgebra: mul!
using TimerOutputs: TimerOutput, @timeit
using Static: Static, StaticBool, False, True
using StaticArrays: SVector
using LinearAlgebra: ⋅
using Base.Cartesian: @nloops, @nref, @ntuple, @nextract, @nexprs
using PrecompileTools: PrecompileTools, @setup_workload, @compile_workload

include("Kernels/Kernels.jl")

using .Kernels:
    Kernels,
    AbstractKernel,
    AbstractKernelData,
    HalfSupport,
    EvaluationMode, Direct, FastApproximation,
    GaussianKernel,
    BSplineKernel,
    KaiserBesselKernel,
    BackwardsKaiserBesselKernel,
    gridstep,
    init_fourier_coefficients!,
    fourier_coefficients

export
    PlanNUFFT,
    HalfSupport,
    Direct, FastApproximation,
    GaussianKernel,
    BSplineKernel,
    KaiserBesselKernel,
    BackwardsKaiserBesselKernel,
    False, True,
    CPU,  # from KernelAbstractions
    set_points!,
    exec_type1!,
    exec_type2!

default_kernel(::KA.Backend) = BackwardsKaiserBesselKernel()
default_kernel() = default_kernel(CPU())

# Default to direct kernel evaluation (can be overridden by specific backends like CUDA).
default_kernel_evalmode(::KA.Backend) = FastApproximation()

default_block_size(::Dims, ::CPU) = 4096  # in number of linear elements
default_block_size(::Dims, ::GPU) = 1024  # except in 2D and 3D (see below)

# TODO: adapt this based on size of shared memory and on element type T (and padding 2M - 1)?
default_block_size(::Dims{2}, ::GPU) = (32, 32)
default_block_size(::Dims{3}, ::GPU) = (16, 16, 4)  # tuned on A100 with 256³ non-oversampled grid, σ = 2 and m = HalfSupport(4)

function default_workgroupsize(backend, ndrange::Dims)
    # This currently assumes 1024 available threads, which might fail in some GPUs.
    KA.default_cpu_workgroupsize(ndrange)
end

# Case of 1D kernels on the GPU (typically, kernels which iterate over non-uniform points).
default_workgroupsize(::GPU, ndrange::Dims{1}) = (1024,)

# This is used at several places instead of getindex (inside of a `map`) to make sure that
# the @inbounds is applied.
inbounds_getindex(v, i) = @inbounds v[i]

include("gpu_common.jl")  # common functions for GPU kernels
include("blocking.jl")
include("plan.jl")
include("set_points.jl")

include("spreading/cpu_nonblocked.jl")
include("spreading/cpu_blocked.jl")
include("spreading/gpu.jl")

include("interpolation/cpu_nonblocked.jl")
include("interpolation/cpu_blocked.jl")
include("interpolation/gpu.jl")

include("abstractNFFTs.jl")

function check_nufft_uniform_data(p::PlanNUFFT, ûs_all::NTuple{C, AbstractArray{<:Complex}}) where {C}
    (; ks,) = p.data
    Nc = ntransforms(p)
    C == Nc || throw(DimensionMismatch(lazy"wrong amount of arrays (expected a tuple of $Nc arrays)"))
    D = length(ks)  # number of dimensions
    Nk_expected = map(length, ks)
    for ûs ∈ ûs_all
        ndims(ûs) == D || throw(DimensionMismatch(lazy"wrong dimensions of array (expected $D-dimensional array)"))
        size(ûs) == Nk_expected || throw(DimensionMismatch(lazy"wrong dimensions of array (expected dimensions $Nk_expected)"))
    end
    nothing
end

function check_nufft_nonuniform_data(p::PlanNUFFT, vp_all::NTuple{C, AbstractVector}) where {C}
    Nc = ntransforms(p)
    C == Nc || throw(DimensionMismatch(lazy"wrong amount of data vectors (expected a tuple of $Nc vectors)"))
    Np = length(p.points)
    for vp ∈ vp_all
        Nv = length(vp)
        Nv == Np || throw(DimensionMismatch(lazy"wrong length of data vector (it should match the number of points $Np, got length $Nv)"))
    end
    nothing
end

@kernel function fill_with_zeros_kernel!(us::NTuple)
    I = @index(Global, Linear)
    for u ∈ us
        @inbounds u[I] = zero(eltype(u))
    end
    nothing
end

"""
    exec_type1!(ûs::AbstractArray{<:Complex}, p::PlanNUFFT, vp::AbstractVector{<:Number})
    exec_type1!(ûs::NTuple{N, AbstractArray{<:Complex}}, p::PlanNUFFT, vp::NTuple{N, AbstractVector{<:Number}})

Perform type-1 NUFFT (from non-uniform points to uniform grid).

Here `vp` contains the input values at non-uniform points.
The result of the transform is written into `ûs`.

One first needs to set the non-uniform points using [`set_points!`](@ref).

To perform multiple transforms at once, both `vp` and `ûs` should be a tuple of arrays (second variant above).
Note that this requires a plan initialised with `ntransforms = Val(N)` (see [`PlanNUFFT`](@ref)).

See also [`exec_type2!`](@ref).
"""
function exec_type1! end

function exec_type1!(ûs_k::NTuple{C, AbstractArray{<:Complex}}, p::PlanNUFFT, vp::NTuple{C}) where {C}
    (; backend, points, kernels, data, blocks, index_map,) = p
    (; us,) = data
    timer = get_timer_nowarn(p)

    @timeit timer "Execute type 1" begin
        check_nufft_uniform_data(p, ûs_k)
        check_nufft_nonuniform_data(p, vp)

        @timeit timer "(0) Fill with zeros" let
            local ndrange = size(us[1])
            local workgroupsize = default_workgroupsize(backend, ndrange)
            local kernel! = fill_with_zeros_kernel!(backend, workgroupsize, ndrange)
            kernel!(us)
            maybe_synchronise(p)
        end

        @timeit timer "(1) Spreading" begin
            spread_from_points!(backend, blocks, kernels, p.kernel_evalmode, us, points, vp)
            maybe_synchronise(p)
        end

        @timeit timer "(2) Forward FFT" begin
            ûs = _type1_fft!(data)
            maybe_synchronise(p)
        end

        @timeit timer "(3) Deconvolution" begin
            T = real(eltype(first(us)))
            normfactor::T = prod(N -> 2π / N, size(first(us)))  # FFT normalisation factor
            ϕ̂s = map(fourier_coefficients, kernels)
            copy_deconvolve_to_non_oversampled!(backend, ûs_k, ûs, index_map, ϕ̂s, normfactor)  # truncate to original grid + normalise
            maybe_synchronise(p)
        end
    end

    ûs_k
end

# Case of a single transform
function exec_type1!(ûs_k::AbstractArray{<:Complex}, p::PlanNUFFT, vp)
    exec_type1!((ûs_k,), p, (vp,))
    ûs_k
end

function _type1_fft!(data::RealNUFFTData)
    (; us, ûs, plan_fw,) = data
    for (u, û) ∈ zip(us, ûs)
        mul!(û, plan_fw, u)  # perform r2c FFT
    end
    ûs
end

function _type1_fft!(data::ComplexNUFFTData)
    (; us, plan_fw,) = data
    for u ∈ us
        plan_fw * u   # perform in-place c2c FFT
    end
    us
end

"""
    exec_type2!(vp::AbstractVector{<:Number}, p::PlanNUFFT, ûs::AbstractArray{<:Complex})
    exec_type2!(vp::NTuple{N, AbstractVector{<:Number}}, p::PlanNUFFT, ûs::NTuple{N, AbstractArray{<:Complex}})

Perform type-2 NUFFT (from uniform grid to non-uniform points).

Here `ûs` contains the input coefficients in the uniform grid.
The result of the transform at non-uniform points is written into `vp`.

One first needs to set the non-uniform points using [`set_points!`](@ref).

To perform multiple transforms at once, both `vp` and `ûs` should be a tuple of arrays (second variant above).
Note that this requires a plan initialised with `ntransforms = Val(N)` (see [`PlanNUFFT`](@ref)).

See also [`exec_type1!`](@ref).
"""
function exec_type2! end

function exec_type2!(vp::NTuple{C, AbstractVector}, p::PlanNUFFT, ûs_k::NTuple{C, AbstractArray{<:Complex}}) where {C}
    (; backend, points, kernels, data, blocks, index_map,) = p
    (; us,) = data
    timer = get_timer_nowarn(p)

    @timeit timer "Execute type 2" begin
        check_nufft_uniform_data(p, ûs_k)
        check_nufft_nonuniform_data(p, vp)
        ûs = output_field(data)

        # Start by zeroing-out the whole output arrays.
        # This can actually be quite costly for big transforms.
        #
        # NOTE: In fact we just need to zero-out the oversampled region (to get zero-padding), but
        # that's more difficult to do and might even be more expensive in multiple dimensions,
        # since that region is not contiguous.
        #
        # TODO: specialise and optimise for 1D case? In that case it might actually be worth
        # it to zero-out only the oversampled region.
        @timeit timer "(0) Fill with zeros" begin
            local ndrange = size(ûs[1])
            local workgroupsize = default_workgroupsize(backend, ndrange)
            local kernel! = fill_with_zeros_kernel!(backend, workgroupsize, ndrange)
            kernel!(ûs)
            maybe_synchronise(p)
        end

        @timeit timer "(1) Deconvolution" begin
            ϕ̂s = map(fourier_coefficients, kernels)
            copy_deconvolve_to_oversampled!(backend, ûs, ûs_k, index_map, ϕ̂s)
            maybe_synchronise(p)
        end

        @timeit timer "(2) Backward FFT" begin
            _type2_fft!(data)
            maybe_synchronise(p)
        end

        @timeit timer "(3) Interpolation" begin
            interpolate!(backend, blocks, kernels, p.kernel_evalmode, vp, us, points)
            maybe_synchronise(p)
        end
    end

    vp
end

# Case of a single transform
function exec_type2!(vp::AbstractVector, p::PlanNUFFT, ûs_k::AbstractArray{<:Complex})
    exec_type2!((vp,), p, (ûs_k,))
end

function _type2_fft!(data::RealNUFFTData)
    (; us, ûs, plan_bw,) = data
    for (u, û) ∈ zip(us, ûs)
        # TODO: can we avoid big GPU allocation on CUDA.jl? (https://github.com/JuliaGPU/CUDA.jl/issues/2249)
        mul!(u, plan_bw, û)  # perform inverse r2c FFT
    end
    us
end

function _type2_fft!(data::ComplexNUFFTData)
    (; us, plan_bw,) = data
    for u ∈ us
        plan_bw * u  # perform in-place inverse c2c FFT
    end
    us
end

# Create index mapping allowing to go from oversampled to non-oversampled wavenumbers.
# If `fftshift = true`, the output wavenumbers are in increasing order (as opposed to FFTW order, which starts at k = 0).
function non_oversampled_indices!(
        indmap::AbstractVector, ks::AbstractVector, ax::AbstractUnitRange;
        fftshift = false,
    )
    @assert length(indmap) == length(ks) ≤ length(ax)
    Nk = length(ks)
    r2c = last(ks) > 0  # true if real-to-complex transform is performed in this dimension
    if r2c
        # Note: copyto! seems to use scalar indexing on GPUs, and thus fails.
        @views indmap[:] .= ax[begin:(begin + Nk - 1)]
    elseif iseven(Nk)
        h = Nk ÷ 2
        if fftshift
            @views indmap[begin:(begin + h - 1)] .= ax[(end - h + 1):end]  # k < 0
            @views indmap[(begin + h):end] .= ax[begin:(begin + h - 1)]    # k ≥ 0
        else
            @views indmap[begin:(begin + h - 1)] .= ax[begin:(begin + h - 1)]  # k ≥ 0
            @views indmap[(begin + h):end] .= ax[(end - h + 1):end]  # k < 0
        end
    else
        h = (Nk - 1) ÷ 2
        if fftshift
            @views indmap[begin:(begin + h - 1)] .= ax[(end - h + 1):end]  # k < 0
            @views indmap[(begin + h):end] .= ax[begin:(begin + h)]        # k ≥ 0
        else
            @views indmap[begin:(begin + h)] .= ax[begin:(begin + h)]
            @views indmap[(begin + h + 1):end] .= ax[(end - h + 1):end]
        end
    end
    indmap
end

function copy_deconvolve_to_non_oversampled!(
        ::CPU, ŵs_all::NTuple{C}, ûs_all::NTuple{C}, index_map, ϕ̂s, normfactor,
    ) where {C}
    @assert C > 0
    inds_out = axes(first(ŵs_all))
    @assert inds_out == map(eachindex, index_map)

    # Split indices (and everything else) in order to parallelise along the last dimension.
    inds_front = Base.front(inds_out)  # indices over dimensions 1, ..., N - 1
    inds_last = last(inds_out)         # indices over dimension N
    index_map_front, index_map_last = Base.front(index_map), last(index_map)
    ϕ̂s_front, ϕ̂s_last = Base.front(ϕ̂s), last(ϕ̂s)

    Threads.@threads :static for i_last ∈ inds_last
        @inbounds begin
            j_last = index_map_last[i_last]
            ϕ̂_last = ϕ̂s_last[i_last]
            for I_front ∈ CartesianIndices(inds_front)
                I = CartesianIndex(I_front, i_last)
                js_front = map(inbounds_getindex, index_map_front, Tuple(I_front))
                ϕ̂_front = map(inbounds_getindex, ϕ̂s_front, Tuple(I_front))
                β = normfactor / (prod(ϕ̂_front) * ϕ̂_last)   # deconvolution + FFT normalisation factor
                for (ŵs, ûs) ∈ zip(ŵs_all, ûs_all)
                    ŵs[I] = β * ûs[js_front..., j_last]
                end
            end
        end
    end

    ŵs_all
end

@kernel function copy_deconvolve_to_non_oversampled_kernel!(
        ŵs_all::NTuple{C}, @Const(ûs_all::NTuple{C}), @Const(index_map), @Const(ϕ̂s), @Const(normfactor),
    ) where {C}
    is = @index(Global, NTuple)                 # output index
    js = map(inbounds_getindex, index_map, is)  # input index
    ϕs_local = map(inbounds_getindex, ϕ̂s, is)   # convolution coefficients (one for each Cartesian direction)
    β = normfactor / prod(ϕs_local)
    for n ∈ eachindex(ŵs_all, ûs_all)
        @inbounds ŵs_all[n][is...] = β * ûs_all[n][js...]
    end
    nothing
end

function copy_deconvolve_to_non_oversampled!(
        backend::GPU, ŵs_all::NTuple{C}, ûs_all::NTuple{C}, index_map, ϕ̂s, normfactor,
    ) where {C}
    @assert C > 0
    ndrange = size(first(ŵs_all))  # size of output array (uniform grid, non oversampled)
    workgroupsize = default_workgroupsize(backend, ndrange)
    kernel! = copy_deconvolve_to_non_oversampled_kernel!(backend, workgroupsize, ndrange)
    kernel!(ŵs_all, ûs_all, index_map, ϕ̂s, normfactor)
    ŵs_all
end

function copy_deconvolve_to_oversampled!(
        backend::CPU, ûs_all::NTuple{C, DenseArray}, ŵs_all::NTuple{C}, index_map, ϕ̂s,
    ) where {C}
    @assert C > 0
    inds_out = axes(first(ŵs_all))
    @assert inds_out == map(eachindex, index_map)

    # Split indices (and everything else) in order to parallelise along the last dimension.
    inds_front = Base.front(inds_out)  # indices over dimensions 1, ..., N - 1
    inds_last = last(inds_out)         # indices over dimension N
    index_map_front, index_map_last = Base.front(index_map), last(index_map)
    ϕ̂s_front, ϕ̂s_last = Base.front(ϕ̂s), last(ϕ̂s)

    Threads.@threads :static for i_last ∈ inds_last
        @inbounds begin
            j_last = index_map_last[i_last]
            ϕ̂_last = ϕ̂s_last[i_last]
            for I_front ∈ CartesianIndices(inds_front)
                I = CartesianIndex(I_front, i_last)
                js_front = map(inbounds_getindex, index_map_front, Tuple(I_front))
                ϕ̂_front = map(inbounds_getindex, ϕ̂s_front, Tuple(I_front))
                β = 1 / (prod(ϕ̂_front) * ϕ̂_last)  # deconvolution factor
                for (ŵs, ûs) ∈ zip(ŵs_all, ûs_all)
                    ûs[js_front..., j_last] = β * ŵs[I]
                end
            end
        end
    end

    ûs_all
end

@kernel function copy_deconvolve_to_oversampled_kernel!(
        ûs_all::NTuple{C}, @Const(ŵs_all::NTuple{C}), @Const(index_map), @Const(ϕ̂s),
    ) where {C}
    is = @index(Global, NTuple)                 # input index (on non-oversampled grid)
    js = map(inbounds_getindex, index_map, is)  # output index (on oversampled grid)
    ϕs_local = map(inbounds_getindex, ϕ̂s, is)   # convolution coefficients (one for each Cartesian direction)
    β = 1 / prod(ϕs_local)                      # deconvolution factor
    for n ∈ eachindex(ŵs_all, ûs_all)
        @inbounds ûs_all[n][js...] = β * ŵs_all[n][is...]
    end
    nothing
end

function copy_deconvolve_to_oversampled!(
        backend::GPU, ûs_all::NTuple{C}, ŵs_all::NTuple{C}, index_map, ϕ̂s,
    ) where {C}
    @assert C > 0
    ndrange = size(first(ŵs_all))  # size of input array (uniform grid, non oversampled)
    workgroupsize = default_workgroupsize(backend, ndrange)
    kernel! = copy_deconvolve_to_oversampled_kernel!(backend, workgroupsize, ndrange)
    kernel!(ûs_all, ŵs_all, index_map, ϕ̂s)
    ŵs_all
end

# Precompile a small subset of possible static parameter combinations (m, kernel, T, ndims,
# ntransforms), to avoid huge precompilation times.
@setup_workload let
    kernels = [default_kernel()]  # only precompile for default kernel
    ndims_all = (1, 3)
    ms = map(HalfSupport, (4,))
    # Ts = [Float32, Float64, ComplexF32, ComplexF64]
    Ts = (Float64,)
    σ = 1.25
    fftw_flags = FFTW.ESTIMATE
    for kernel ∈ kernels, m ∈ ms, T ∈ Ts, ndims ∈ ndims_all
        dims = ntuple(_ -> 13, ndims)  # floor(σN) = 16
        Np = 16  # number of non-uniform points
        xs = ntuple(_ -> rand(real(T), Np) .* 2π, ndims)
        Ns = ntuple(ndims) do i
            N = dims[i]
            (i == 1 && T <: Real) ? ((N + 2) >> 1) : N
        end
        C = complex(T)
        for ntrans ∈ (1,)  # number of simultaneous transforms
            ntransforms = Val(ntrans)
            qs = ntuple(_ -> randn(T, Np), ntransforms)
            uhat = ntuple(_ -> Array{C}(undef, Ns), ntransforms)
            @compile_workload begin
                plan = PlanNUFFT(T, dims; ntransforms, fftw_flags, m, σ, kernel)
                set_points!(plan, xs)
                exec_type1!(uhat, plan, qs)
                exec_type2!(qs, plan, uhat)
            end
        end
    end
end

end
