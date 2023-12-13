module NonuniformFFTs

using StructArrays: StructVector
using FFTW: FFTW
using LinearAlgebra: mul!
using Polyester: @batch

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
    init_fourier_coefficients!

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
include("spreading.jl")
include("interpolation.jl")

# Here the element type of `xp` can either be an NTuple{N, <:Real}, an SVector{N, <:Real},
# or anything else which has length `N`.
function set_points!(p::PlanNUFFT{T, N}, xp::AbstractVector) where {T, N}
    (; points,) = p
    type_length(eltype(xp)) == N || throw(DimensionMismatch(lazy"expected $N-dimensional points"))
    resize!(points, length(xp))
    Base.require_one_based_indexing(points)
    @inbounds for (i, x) ∈ enumerate(xp)
        points[i] = to_unit_cell(NTuple{N}(x))  # converts `x` to Tuple if it's an SVector
    end
    p
end

to_unit_cell(x⃗) = map(_to_unit_cell, x⃗)

function _to_unit_cell(x::Real)
    L = oftype(x, 2π)
    while x < 0
        x += L
    end
    while x ≥ L
        x -= L
    end
    x
end

type_length(::Type{T}) where {T} = length(T)  # usually for SVector
type_length(::Type{<:NTuple{N}}) where {N} = N

function set_points!(p::PlanNUFFT{T, N}, xp::NTuple{N, AbstractVector}) where {T, N}
    set_points!(p, StructVector(xp))
end

# 1D case
function set_points!(p::PlanNUFFT{T, 1}, xp::AbstractVector{<:Real}) where {T}
    set_points!(p, StructVector((xp,)))
end

function check_nufft_uniform_data(p::PlanNUFFT, ûs_k::AbstractArray{<:Complex})
    (; ks,) = p.data
    D = length(ks)  # number of dimensions
    ndims(ûs_k) == D || throw(DimensionMismatch(lazy"wrong dimensions of output array (expected $D-dimensional array)"))
    Nk_expected = map(length, ks)
    size(ûs_k) == Nk_expected || throw(DimensionMismatch(lazy"wrong dimensions of output array (expected dimensions $Nk_expected)"))
    nothing
end

function exec_type1!(ûs_k::AbstractArray{<:Complex}, p::PlanNUFFT, charges)
    (; points, kernels, data, blocks,) = p
    (; us, ks,) = data
    check_nufft_uniform_data(p, ûs_k)
    fill!(us, zero(eltype(us)))
    if isempty(blocks.buffers)
        spread_from_points!(kernels, us, points, charges)  # single-threaded case?
    else
        spread_from_points_blocked!(kernels, blocks, us, points, charges)
    end
    ûs = _type1_fft!(data)
    T = real(eltype(us))
    normfactor::T = prod(N -> 2π / N, size(us))  # FFT normalisation factor
    ϕ̂s = map(init_fourier_coefficients!, kernels, ks)  # this takes time only the first time it's called
    copy_deconvolve_to_non_oversampled!(ûs_k, ûs, ks, ϕ̂s, normfactor)  # truncate to original grid + normalise
    ûs_k
end

function _type1_fft!(data::RealNUFFTData)
    (; us, ûs, plan_fw,) = data
    mul!(ûs, plan_fw, us)  # perform r2c FFT
    ûs
end

function _type1_fft!(data::ComplexNUFFTData)
    (; us, plan_fw,) = data
    plan_fw * us   # perform in-place c2c FFT
    us
end

function exec_type2!(vp::AbstractVector, p::PlanNUFFT, ûs_k::AbstractArray{<:Complex})
    (; points, kernels, data,) = p
    (; us, ks,) = data
    check_nufft_uniform_data(p, ûs_k)
    ϕ̂s = map(init_fourier_coefficients!, kernels, ks)  # this takes time only the first time it's called
    _type2_copy_and_fft!(ûs_k, ϕ̂s, data)
    interpolate!(kernels, vp, us, points)
    vp
end

function _type2_copy_and_fft!(ûs_k, ϕ̂s, data::RealNUFFTData)
    (; us, ûs, ks, plan_bw,) = data
    copy_deconvolve_to_oversampled!(ûs, ûs_k, ks, ϕ̂s)
    mul!(us, plan_bw, ûs)  # perform inverse r2c FFT
    nothing
end

function _type2_copy_and_fft!(ûs_k, ϕ̂s, data::ComplexNUFFTData)
    (; us, ks, plan_bw,) = data
    copy_deconvolve_to_oversampled!(us, ûs_k, ks, ϕ̂s)
    plan_bw * us  # perform in-place inverse c2c FFT
    nothing
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
    for (I, J) ∈ zip(inds_k, inds)
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
    for (I, J) ∈ zip(inds_k, inds)
        ϕ̂ = map(getindex, ϕ̂s, Tuple(I))  # Fourier coefficient of kernel
        ûs[J...] = ûs_k[I] / prod(ϕ̂)
    end
    ûs_k
end

end
