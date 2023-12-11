module NonuniformFFTs

# TODO
# - try piecewise polynomial approximation?

using StructArrays: StructVector
using FFTW: FFTW
using LinearAlgebra: mul!

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

include("spreading.jl")
include("interpolation.jl")
include("convolution.jl")

# TODO
# - allow complex non-uniform values?
struct PlanNUFFT{
        T <: AbstractFloat, N, M,
        Kernels <: NTuple{N, AbstractKernelData{<:AbstractKernel, M, T}},
        WaveNumbers <: NTuple{N, AbstractVector{T}},
        Points <: StructVector{NTuple{N, T}},
        PlanFFT_fw <: FFTW.Plan{T},
        PlanFFT_bw <: FFTW.Plan{Complex{T}},
    }
    kernels :: Kernels
    ks      :: WaveNumbers  # wavenumbers in *non-oversampled* Fourier grid
    σ       :: T            # oversampling factor (≥ 1)
    points  :: Points       # non-uniform points (real values)
    us      :: Array{T, N}  # values in oversampled grid
    ûs      :: Array{Complex{T}, N}  # Fourier coefficients in oversampled grid
    plan_fw :: PlanFFT_fw
    plan_bw :: PlanFFT_bw
end

# This constructor is generally not called directly.
function _PlanNUFFT(
        kernels::NTuple{D, <:AbstractKernelData}, σ_wanted, Ns::Dims{D};
        fftw_flags = FFTW.MEASURE,
    ) where {D}
    T = typeof(σ_wanted)
    ks = ntuple(Val(length(Ns))) do i
        N = Ns[i]
        # This assumes L = 2π:
        i == 1 ? FFTW.rfftfreq(N, T(N)) : FFTW.fftfreq(N, T(N))
    end
    # Determine dimensions of oversampled grid.
    Ñs = map(Ns) do N
        # We try to make sure that each dimension is a product of powers of small primes,
        # which is good for FFT performance.
        nextprod((2, 3, 5), floor(Int, σ_wanted * N))
    end
    σ::T = maximum(Ñs ./ Ns)  # actual oversampling factor
    points = StructVector(ntuple(_ -> T[], Val(D)))
    us = Array{T}(undef, Ñs)
    dims_out = (Ñs[1] ÷ 2 + 1, Base.tail(Ñs)...)
    ûs = Array{Complex{T}}(undef, dims_out)
    plan_fw = FFTW.plan_rfft(us; flags = fftw_flags)
    plan_bw = FFTW.plan_brfft(ûs, size(us, 1); flags = fftw_flags)
    PlanNUFFT(kernels, ks, σ, points, us, ûs, plan_fw, plan_bw)
end

function PlanNUFFT(
        ::Type{T}, Ns::Dims, h::HalfSupport;
        kernel::AbstractKernel = BackwardsKaiserBesselKernel(),
        σ::Real = real(T)(2), kws...,
    ) where {T <: AbstractFloat}
    let σ = T(σ)
        L = T(2π)  # assume 2π period
        kernels = map(Ns) do N
            Δx̃ = L / N / σ
            Kernels.optimal_kernel(kernel, h, Δx̃, σ)
        end
        _PlanNUFFT(kernels, σ, Ns; kws...)
    end
end

# This constructor relies on constant propagation to make the output fully inferred.
function PlanNUFFT(::Type{T}, Ns::Dims; m::Integer = 8, kws...) where {T}
    h = HalfSupport(m)
    PlanNUFFT(T, Ns, h; kws...)
end

# 1D case
function PlanNUFFT(::Type{T}, N::Integer, args...; kws...) where {T <: AbstractFloat}
    PlanNUFFT(T, (N,), args...; kws...)
end

# Alternative constructor: use default floating point type.
function PlanNUFFT(N::Union{Integer, Dims}, args...; kws...)
    PlanNUFFT(Float64, N, args...; kws...)
end

function set_points!(p::PlanNUFFT{T, N}, xp::AbstractVector{<:NTuple{N}}) where {T, N}
    (; points,) = p
    resize!(points, length(xp))
    Base.require_one_based_indexing(points)
    @inbounds for (i, x) ∈ enumerate(xp)
        points[i] = x
    end
    p
end

function set_points!(p::PlanNUFFT{T, N}, xp::NTuple{N, AbstractVector}) where {T, N}
    set_points!(p, StructVector(xp))
end

# 1D case
function set_points!(p::PlanNUFFT{T, 1}, xp::AbstractVector{<:Real}) where {T}
    set_points!(p, StructVector((xp,)))
end

function check_nufft_uniform_data(p::PlanNUFFT, ûs_k::AbstractArray{<:Complex})
    (; ks,) = p
    D = length(ks)  # number of dimensions
    ndims(ûs_k) == D || throw(DimensionMismatch(lazy"wrong dimensions of output array (expected $D-dimensional array)"))
    Nk_expected = map(length, ks)
    size(ûs_k) == Nk_expected || throw(DimensionMismatch(lazy"wrong dimensions of output array (expected dimensions $Nk_expected)"))
    nothing
end

function exec_type1!(ûs_k::AbstractArray{<:Complex}, p::PlanNUFFT, charges)
    (; points, kernels, us, ûs, ks, plan_fw,) = p
    check_nufft_uniform_data(p, ûs_k)
    fill!(us, zero(eltype(us)))
    spread_from_points!(kernels, us, points, charges)
    mul!(ûs, plan_fw, us)         # perform FFT
    T = real(eltype(us))
    normfactor::T = prod(N -> 2π / N, size(us))  # FFT normalisation factor
    ϕ̂s = map(init_fourier_coefficients!, kernels, ks)  # this takes time only the first time it's called
    copy_deconvolve_to_non_oversampled!(ûs_k, ûs, ks, ϕ̂s, normfactor)  # truncate to original grid + normalise
    ûs_k
end

function exec_type2!(vp::AbstractVector, p::PlanNUFFT, ûs_k::AbstractArray{<:Complex})
    (; points, kernels, us, ûs, ks, plan_bw,) = p
    check_nufft_uniform_data(p, ûs_k)
    ϕ̂s = map(init_fourier_coefficients!, kernels, ks)  # this takes time only the first time it's called
    copy_deconvolve_to_oversampled!(ûs, ûs_k, ks, ϕ̂s)
    mul!(us, plan_bw, ûs)  # perform inverse FFT
    interpolate!(kernels, vp, us, points)
    vp
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
