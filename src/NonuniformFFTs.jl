module NonuniformFFTs

using StructArrays: StructVector
using FFTW: FFTW
using LinearAlgebra: mul!
using TimerOutputs: TimerOutput, @timeit
using Static: Static, StaticBool, False, True
using PrecompileTools: PrecompileTools, @setup_workload, @compile_workload

include("Kernels/Kernels.jl")

using .Kernels:
    Kernels,
    AbstractKernel,
    AbstractKernelData,
    HalfSupport,
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
    GaussianKernel,
    BSplineKernel,
    KaiserBesselKernel,
    BackwardsKaiserBesselKernel,
    False, True,
    set_points!,
    exec_type1!,
    exec_type2!

default_kernel() = BackwardsKaiserBesselKernel()

# This is used at several places instead of getindex (inside of a `map`) to be sure that the
# @inbounds is applied.
inbounds_getindex(v, i) = @inbounds v[i]

include("sorting.jl")
include("blocking.jl")
include("plan.jl")
include("set_points.jl")
include("spreading.jl")
include("interpolation.jl")

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
    (; points, kernels, data, blocks, index_map, timer,) = p
    (; us, ks,) = data
    @timeit timer "Execute type 1" begin
        check_nufft_uniform_data(p, ûs_k)
        check_nufft_nonuniform_data(p, vp)
        for u ∈ us
            fill!(u, zero(eltype(u)))
        end
        @timeit timer "Spreading" if with_blocking(blocks)
            spread_from_points_blocked!(kernels, blocks, us, points, vp)
        else
            spread_from_points!(kernels, us, points, vp)  # single-threaded case?
        end
        @timeit timer "Forward FFT" ûs = _type1_fft!(data)
        @timeit timer "Deconvolution" begin
            T = real(eltype(first(us)))
            normfactor::T = prod(N -> 2π / N, size(first(us)))  # FFT normalisation factor
            ϕ̂s = map(fourier_coefficients, kernels)
            copy_deconvolve_to_non_oversampled!(ûs_k, ûs, index_map, ϕ̂s, normfactor)  # truncate to original grid + normalise
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
    (; points, kernels, data, blocks, index_map, timer,) = p
    (; us, ks,) = data
    @timeit timer "Execute type 2" begin
        check_nufft_uniform_data(p, ûs_k)
        check_nufft_nonuniform_data(p, vp)
        @timeit timer "Deconvolution" begin
            ϕ̂s = map(fourier_coefficients, kernels)
            ûs = output_field(data)
            copy_deconvolve_to_oversampled!(ûs, ûs_k, index_map, ϕ̂s)
        end
        @timeit timer "Backward FFT" _type2_fft!(data)
        @timeit timer "Interpolate" begin
            if with_blocking(blocks)
                interpolate_blocked!(kernels, blocks, vp, us, points)
            else
                interpolate!(kernels, vp, us, points)
            end
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
function non_oversampled_indices!(
        indmap::AbstractVector, ks::AbstractVector, ax::AbstractUnitRange,
    )
    @assert length(indmap) == length(ks) ≤ length(ax)
    Nk = length(ks)
    r2c = last(ks) > 0  # true if real-to-complex transform is performed in this dimension
    if r2c
        copyto!(indmap, ax[begin:(begin + Nk - 1)])
    elseif iseven(Nk)
        h = Nk ÷ 2
        @views copyto!(indmap[begin:(begin + h - 1)], ax[begin:(begin + h - 1)])
        @views copyto!(indmap[(begin + h):end], ax[(end - h + 1):end])
    else
        h = (Nk - 1) ÷ 2
        @views copyto!(indmap[begin:(begin + h)], ax[begin:(begin + h)])
        @views copyto!(indmap[(begin + h + 1):end], ax[(end - h + 1):end])
    end
    indmap
end

function copy_deconvolve_to_non_oversampled!(
        ŵs_all::NTuple{C}, ûs_all::NTuple{C}, index_map, ϕ̂s, normfactor,
    ) where {C}
    @assert C > 0
    inds_out = axes(first(ŵs_all))
    @assert inds_out == map(eachindex, index_map)

    # Split indices (and everything else) in order to parallelise along the last dimension.
    inds_front = Base.front(inds_out)  # indices over dimensions 1, ..., N - 1
    inds_last = last(inds_out)         # indices over dimension N
    index_map_front, index_map_last = Base.front(index_map), last(index_map)
    ϕ̂s_front, ϕ̂s_last = Base.front(ϕ̂s), last(ϕ̂s)

    Threads.@threads :static for ilast ∈ inds_last
        @inbounds begin
            jlast = index_map_last[ilast]
            ϕ̂last = ϕ̂s_last[ilast]
            for Ifront ∈ CartesianIndices(inds_front)
                I = CartesianIndex(Ifront, ilast)
                js_front = map(inbounds_getindex, index_map_front, Tuple(Ifront))
                ϕ̂_front = map(inbounds_getindex, ϕ̂s_front, Tuple(Ifront))
                β = normfactor / (prod(ϕ̂_front) * ϕ̂last)   # deconvolution + FFT normalisation factor
                for (ŵs, ûs) ∈ zip(ŵs_all, ûs_all)
                    ŵs[I] = β * ûs[js_front..., jlast]
                end
            end
        end
    end

    ŵs_all
end

function copy_deconvolve_to_oversampled!(
        ûs_all::NTuple{C}, ŵs_all::NTuple{C}, index_map, ϕ̂s,
    ) where {C}
    @assert C > 0

    # TODO: is this needed? and can it be optimised, by zeroing out only what's needed?
    inds_oversampled = eachindex(first(ûs_all)) :: AbstractVector  # array uses linear indices
    @assert isone(first(inds_oversampled))  # 1-based indexing
    @assert all(ûs -> eachindex(ûs) === inds_oversampled, ûs_all)  # all arrays have the same indices
    Threads.@threads :static for n ∈ 1:Threads.nthreads()
        a = ((n - 1) * length(inds_oversampled)) ÷ Threads.nthreads()
        b = ((n - 0) * length(inds_oversampled)) ÷ Threads.nthreads()
        checkbounds(inds_oversampled, (a + 1):b)
        for ûs ∈ ûs_all
            p = pointer(ûs, a + 1)
            n = (b - a) * sizeof(eltype(ûs))
            val = zero(Cint)
            ccall(:memset, Ptr{Cvoid}, (Ptr{Cvoid}, Cint, Csize_t), p, val, n)
        end
    end

    inds_out = axes(first(ŵs_all))
    @assert inds_out == map(eachindex, index_map)

    # Split indices (and everything else) in order to parallelise along the last dimension.
    inds_front = Base.front(inds_out)  # indices over dimensions 1, ..., N - 1
    inds_last = last(inds_out)         # indices over dimension N
    index_map_front, index_map_last = Base.front(index_map), last(index_map)
    ϕ̂s_front, ϕ̂s_last = Base.front(ϕ̂s), last(ϕ̂s)

    Threads.@threads :static for ilast ∈ inds_last
        @inbounds begin
            jlast = index_map_last[ilast]
            ϕ̂last = ϕ̂s_last[ilast]
            for Ifront ∈ CartesianIndices(inds_front)
                I = CartesianIndex(Ifront, ilast)
                js_front = map(inbounds_getindex, index_map_front, Tuple(Ifront))
                ϕ̂_front = map(inbounds_getindex, ϕ̂s_front, Tuple(Ifront))
                β = 1 / (prod(ϕ̂_front) * ϕ̂last)  # deconvolution factor
                for (ŵs, ûs) ∈ zip(ŵs_all, ûs_all)
                    ûs[js_front..., jlast] = β * ŵs[I]
                end
            end
        end
    end

    ûs_all
end

# Precompilation
@setup_workload let
    kernels = [default_kernel()]  # only precompile for default kernel
    ndims_all = 1:3
    ms = map(HalfSupport, 2:2:8)
    # Ts = [Float32, Float64, ComplexF32, ComplexF64]
    Ts = [Float64, ComplexF64]
    σ = 1.25
    for kernel ∈ kernels, m ∈ ms, T ∈ Ts, ndims ∈ ndims_all
        dims = ntuple(_ -> 12, ndims)  # σN = 16
        Np = 16  # number of non-uniform points
        xs = ntuple(_ -> rand(real(T), Np) .* 2π, ndims)
        Ns = ntuple(ndims) do i
            N = dims[i]
            (i == 1 && T <: Real) ? ((N + 2) >> 1) : N
        end
        C = complex(T)
        for ntrans ∈ (1, ndims)  # number of simultaneous transforms
            ntransforms = Val(ntrans)
            qs = ntuple(_ -> randn(T, Np), ntransforms)
            uhat = ntuple(_ -> Array{C}(undef, Ns), ntransforms)
            @compile_workload begin
                plan = PlanNUFFT(T, dims; ntransforms, m, σ, kernel)
                set_points!(plan, xs)
                exec_type1!(uhat, plan, qs)
                exec_type2!(qs, plan, uhat)
            end
        end
    end
end

end
