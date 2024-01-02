module NonuniformFFTs

using StructArrays: StructVector
using FFTW: FFTW
using LinearAlgebra: mul!
using TimerOutputs: TimerOutput, @timeit

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
    set_points!,
    exec_type1!,
    exec_type2!

include("blocking.jl")
include("plan.jl")
include("set_points.jl")
include("spreading.jl")
include("interpolation.jl")

function check_nufft_uniform_data(p::PlanNUFFT, ûs_all::NTuple{Nc′, AbstractArray{<:Complex}}) where {Nc′}
    (; ks,) = p.data
    Nc = ntransforms(p)
    Nc′ == Nc || throw(DimensionMismatch(lazy"wrong amount of arrays (expected a tuple of $Nc arrays)"))
    D = length(ks)  # number of dimensions
    Nk_expected = map(length, ks)
    for ûs ∈ ûs_all
        ndims(ûs) == D || throw(DimensionMismatch(lazy"wrong dimensions of array (expected $D-dimensional array)"))
        size(ûs) == Nk_expected || throw(DimensionMismatch(lazy"wrong dimensions of array (expected dimensions $Nk_expected)"))
    end
    nothing
end

function exec_type1!(ûs_k::NTuple{<:Any, AbstractArray{<:Complex}}, p::PlanNUFFT, charges)
    (; points, kernels, data, blocks, timer,) = p
    (; us, ks,) = data
    @timeit timer "Execute type 1" begin
        check_nufft_uniform_data(p, ûs_k)
        fill!(us, zero(eltype(us)))
        @timeit timer "Spreading" if with_blocking(blocks)
            spread_from_points_blocked!(kernels, blocks, us, points, charges)
        else
            spread_from_points!(kernels, us, points, charges)  # single-threaded case?
        end
        @timeit timer "Forward FFT" ûs = _type1_fft!(data)
        @timeit timer "Deconvolution" begin
            T = real(eltype(us))
            normfactor::T = prod(N -> 2π / N, size(us))  # FFT normalisation factor
            ϕ̂s = map(fourier_coefficients, kernels)
            for (û, ŵ) ∈ zip(ûs, ûs_k)
                copy_deconvolve_to_non_oversampled!(ŵ, û, ks, ϕ̂s, normfactor)  # truncate to original grid + normalise
            end
        end
    end
    ûs_k
end

# Case of a single transform
function exec_type1!(ûs_k::AbstractArray{<:Complex}, p::PlanNUFFT, charges)
    exec_type1!((ûs_k,), p, charges)
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

function exec_type2!(vp::AbstractVector, p::PlanNUFFT, ûs_k::NTuple{<:Any, AbstractArray{<:Complex}})
    (; points, kernels, data, blocks, timer,) = p
    (; us, ks,) = data
    @timeit timer "Execute type 2" begin
        check_nufft_uniform_data(p, ûs_k)
        @timeit timer "Deconvolution" begin
            ϕ̂s = map(fourier_coefficients, kernels)
            ûs = output_field(data)
            for (û, ŵ) ∈ zip(ûs, ûs_k)
                copy_deconvolve_to_oversampled!(û, ŵ, ks, ϕ̂s)
            end
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
    exec_type2!(vp, p, (ûs_k,))
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

function non_oversampled_indices(ks::AbstractVector, ax::AbstractUnitRange)
    @assert length(ks) ≤ length(ax)
    Nk = length(ks)
    r2c = last(ks) > 0  # true if real-to-complex transform is performed in this dimension
    inds = if r2c
        (ax[begin:(begin + Nk - 1)], ax[1:0])  # include second empty iterator for type stability
    elseif iseven(Nk)
        h = Nk ÷ 2
        (ax[begin:(begin + h - 1)], ax[(end - h + 1):end])
    else
        h = (Nk - 1) ÷ 2
        (ax[begin:(begin + h)], ax[(end - h + 1):end])
    end
    Iterators.flatten(inds)
end

function copy_deconvolve_to_non_oversampled!(ûs_k, ûs, ks, ϕ̂s, normfactor)
    subindices = map(non_oversampled_indices, ks, axes(ûs))
    inds = Iterators.product(subindices...)  # indices of oversampled array
    inds_k = CartesianIndices(ûs_k)    # indices of non-oversampled array
    @inbounds for (I, J) ∈ zip(inds_k, inds)
        ϕ̂ = map(getindex, ϕ̂s, Tuple(I))  # Fourier coefficient of kernel
        ûs_k[I] = ûs[J...] * (normfactor / prod(ϕ̂))
    end
    ûs_k
end

function copy_deconvolve_to_oversampled!(ûs, ûs_k, ks, ϕ̂s)
    fill!(ûs, zero(eltype(ûs)))  # make sure the padding region is set to zero
    # Note: both these indices should have the same lengths.
    subindices = map(non_oversampled_indices, ks, axes(ûs))
    inds = Iterators.product(subindices...)  # indices of oversampled array
    inds_k = CartesianIndices(ûs_k)    # indices of non-oversampled array
    @inbounds for (I, J) ∈ zip(inds_k, inds)
        ϕ̂ = map(getindex, ϕ̂s, Tuple(I))  # Fourier coefficient of kernel
        ûs[J...] = ûs_k[I] / prod(ϕ̂)
    end
    ûs_k
end

end
