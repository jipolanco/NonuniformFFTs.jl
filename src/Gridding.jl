"""
    Gridding

Module for spreading point data onto a grid using smoothing filters.
"""
module Gridding

using StaticArrays: StaticArrays
using Reexport

include("Kernels/Kernels.jl")
@reexport using .Kernels
using .Kernels:
    Kernels,
    AbstractKernel,
    scale,
    gridstep,
    init_fourier_coefficients!

export
    spread_from_point!,
    spread_from_points!,
    deconvolve_fourier!

include("spreading.jl")
include("interpolation.jl")

"""
    deconvolve_fourier!(
        gs::NTuple{D, AbstractKernel},
        [dest::AbstractArray{T, D}],
        src::AbstractArray{T, D},
        ks::NTuple{D, AbstractVector},
    )

Apply deconvolution to field in Fourier space.

The wavenumber vectors included in `ks` are typically obtained from
`AbstractFFTs.(r)fftfreq`.

If `dest` is not passed, then `src` will be overwritten.

As with [`spread_from_point!`](@ref), `dest` and `src` can also be tuples of arrays.
"""
function deconvolve_fourier!(
        gs::NTuple{D, AbstractKernel},
        dst::NTuple{M, AbstractArray{T, D}},
        src::NTuple{M, AbstractArray{T, D}},
        ks::NTuple{D, AbstractVector},
    ) where {D, M, T <: Complex}
    @assert M > 0
    gks = map(init_fourier_coefficients!, gs, ks)  # this takes time only the first time it's called
    inds = CartesianIndices(first(dst))
    @inbounds for I ∈ inds
        g⃗ = map(getindex, gks, Tuple(I))
        gdiv = 1 / prod(g⃗)
        for (u, v) ∈ zip(dst, src)
            u[I] = v[I] * gdiv
        end
    end
    dst
end

# TODO do we need this??
function convolve_fourier!(
        gs::NTuple{D, AbstractKernel},
        dst::NTuple{M, AbstractArray{T, D}},
        src::NTuple{M, AbstractArray{T, D}},
        ks::NTuple{D},
    ) where {D, M, T <: Complex}
    @assert M > 0
    gks = map(init_fourier_coefficients!, gs, ks)  # this takes time only the first time it's called
    inds = CartesianIndices(first(dst))
    @inbounds for I ∈ inds
        g⃗ = map(getindex, gks, Tuple(I))
        gmul = prod(g⃗)
        for (u, v) ∈ zip(dst, src)
            u[I] = v[I] * gmul
        end
    end
    dst
end

for f ∈ (:deconvolve_fourier!, :convolve_fourier!)
    @eval begin
        # Scalar field version
        $f(
            gs::NTuple{D}, dst::AbstractArray, src::AbstractArray, ks::NTuple{D},
        ) where {D} = only($f(gs, (dst,), (src,), ks))

        # 1D version
        $f(gx::AbstractKernel, dst, src, kx::AbstractVector) =
            $f((gx,), dst, src, (kx,))

        # In-place version
        $f(gs, src, ks) = $f(gs, src, src, ks)
    end
end

end
