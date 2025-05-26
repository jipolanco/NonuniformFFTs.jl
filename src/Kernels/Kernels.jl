module Kernels

using KernelAbstractions: KernelAbstractions as KA
using Adapt: Adapt, adapt

export HalfSupport, EvaluationMode, Direct, FastApproximation

struct HalfSupport{M} end
HalfSupport(M) = HalfSupport{M}()
half_support(::HalfSupport{M}) where {M} = M::Int

Base.show(io::IO, ::HalfSupport{M}) where {M} = print(io, "HalfSupport($M)")

abstract type EvaluationMode end

"""
    Direct <: Kernels.EvaluationMode

Directly evaluate kernels using their definition.
"""
struct Direct <: EvaluationMode end

"""
    FastApproximation <: Kernels.EvaluationMode

Use a fast approximation or algorithm to efficiently evaluate a kernel.

The evaluation type depends on the actual kernel:

- for [`KaiserBesselKernel`](@ref) and [`BackwardsKaiserBesselKernel`](@ref), this uses a
  fast piecewise polynomial approximation heavily inspired from FINUFFT;

- for [`GaussianKernel`](@ref), this uses the fast Gaussian gridding algorithm proposed by
  Greengard & Lee (SIAM Rev. 2004);

- for [`BSplineKernel`](@ref), this is currently the same as [`Direct`](@ref) evaluation.

In the first two cases, this is generally faster than [`Direct`](@ref) evaluation for CPU
transforms.

On GPUs this is not necessarily the case, and results may depend on the actual GPU used.
For example, on an Nvidia A100, `Direct` evaluation of the `KaiserBesselKernel` (which
involves Bessel functions) seems to beat polynomial approximation, and is also a bit faster
than the `BackwardsKaiserBesselKernel` (involving `sinh` functions).
"""
struct FastApproximation <: EvaluationMode end

"""
    AbstractKernel

Abstract type representing a smoothing kernel function.
"""
abstract type AbstractKernel end

"""
    AbstractKernelData{K <: AbstractKernel, M, T}

Abstract type representing an object which contains data associated to a kernel.

Type parameters are a smoothing kernel `K` with half-support `M` (an integer value) and
element type `T`.
"""
abstract type AbstractKernelData{K <: AbstractKernel, M, T <: AbstractFloat} end

"""
    half_support(g::AbstractKernelData{K, M}) -> M

Returns the half-support `M` of the kernel.

This is half the number of grid points where the kernel is evaluated at each
convolution.
"""
half_support(::AbstractKernelData{K, M}) where {K, M} = M::Int

"""
    fourier_coefficients(g::AbstractKernelData) -> AbstractVector

Returns vector with Fourier coefficients associated to the kernel.

The Fourier coefficients must first be precomputed at the chosen wavenumbers
using [`init_fourier_coefficients!`](@ref).
Otherwise, this just returns an empty vector.
"""
fourier_coefficients(g::AbstractKernelData) = g.gk

gridstep(g::AbstractKernelData) = g.Δx

"""
    init_fourier_coefficients!(g::AbstractKernelData, ks::AbstractVector) -> AbstractVector

Precompute Fourier coefficients associated to kernel `g` at wavenumbers `ks`.

Returns a vector with the computed coefficients.

If the coefficient vector was already computed in a previous call, then this
function just returns the coefficients.
"""
function init_fourier_coefficients!(g::AbstractKernelData, ks::AbstractVector)
    gk = fourier_coefficients(g)
    Nk = length(ks)
    Nk == length(gk) && return gk  # assume coefficients were already computed
    resize!(gk, Nk)
    @assert eachindex(gk) == eachindex(ks)
    f = evaluate_fourier_func(g)
    @. gk = f(ks)
    gk
end

# Assign a cell index to a location `x`. This assumes 0 ≤ x < 2π.
# This also returns x / Δx to avoid recomputing it later.
@inline function point_to_cell(x, Δx)
    r = x / Δx
    i = unsafe_trunc(Int, r)  # assumes r ≥ 0
    # Increment by 1 (for one-based indexing), except to avoid possible roundoff errors when x
    # is very close (but slightly smaller) to i * Δx.
    i += (i * Δx ≤ x)  # this is almost always true (so we increment by 1)
    # @assert (i - 1) * Δx ≤ x < i * Δx
    i, r
end

# Note: evaluate_kernel_func generates a function which is callable from GPU kernels.
@inline evaluate_kernel(g::AbstractKernelData, x₀) = evaluate_kernel_func(g)(x₀)

@inline function evaluate_kernel_direct(g::AbstractKernelData, x)
    Δx = gridstep(g)
    i, r = point_to_cell(x, Δx)
    values = _evaluate_kernel_direct(g, i, r)
    (; i, values,)
end

@inline evaluate_kernel(::FastApproximation, g::AbstractKernelData, x) = evaluate_kernel(g, x)
@inline evaluate_kernel(::Direct, g::AbstractKernelData, x) = evaluate_kernel_direct(g, x)

@inline function kernel_indices(i, ::AbstractKernelData{K, M}, args...) where {K, M}
    kernel_indices(i, HalfSupport(M), args...)
end

# Takes into account periodic wrapping.
# This is equivalent to calling mod1(j, N) for each j, but much much faster.
# We assume the central index `i` is in 1:N and that M < N / 2.
function kernel_indices(i, ::HalfSupport{M}, N::Integer) where {M}
    L = 2M
    j = i - M
    j = ifelse(j < 0, j + N, j)
    jref = Ref(j)
    ntuple(Val(L)) do _
        @inline
        jref[] = ifelse(jref[] == N, 1, jref[] + 1)
        jref[]
    end
end

# This variant can be used when periodic wrapping is not needed.
# (Used when doing block partitioning for parallelisation using threads.)
function kernel_indices(i, ::HalfSupport{M}) where {M}
    (i - M + 1):(i + M)
end

# Returns evaluation points around the normalised location X ∈ [0, 1/M).
# Note that points are returned in decreasing order.
function evaluation_points(::Val{M}, X) where {M}
    ntuple(Val(2M)) do j
        X + (M - j) / M  # in [-1, 1]
    end
end

include("piecewise_polynomial.jl")

include("gaussian.jl")
include("bspline.jl")
include("kaiser_bessel.jl")
include("kaiser_bessel_backwards.jl")

end
