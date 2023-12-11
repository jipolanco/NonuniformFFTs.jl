module Kernels

export HalfSupport

struct HalfSupport{M} end
HalfSupport(M) = HalfSupport{M}()
half_support(::HalfSupport{M}) where {M} = M::Int

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
    @inbounds for (i, k) ∈ pairs(ks)
        gk[i] = evaluate_fourier(g, k)
    end
    gk
end

@inline function evaluate_kernel(g::AbstractKernelData, x₀)
    dx = gridstep(g)
    i = floor(Int, x₀ / dx) + 1  # such that xs[i] ≤ x₀ < xs[i + 1]
    evaluate_kernel(g, x₀, i)
end

# Takes into account periodic wrapping.
# This is equivalent to calling mod1(j, N) for each j, but much much faster.
# We assume the central index `i` is in 1:N and that M < N / 2.
function kernel_indices(i, ::AbstractKernelData{K, M}, N::Integer) where {K, M}
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
