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
    inds = CartesianIndices(first(src))
    @inbounds for I ∈ inds
        g⃗ = map(getindex, gks, Tuple(I))
        gdiv = 1 / prod(g⃗)
        for (u, v) ∈ zip(dst, src)
            u[I] = v[I] * gdiv
        end
    end
    dst
end

# Scalar field version
deconvolve_fourier!(
    gs::NTuple{D}, dst::AbstractArray, src::AbstractArray, ks::NTuple{D},
) where {D} = only(deconvolve_fourier!(gs, (dst,), (src,), ks))

# 1D version
deconvolve_fourier!(gx::AbstractKernel, dst, src, kx::AbstractVector) =
    deconvolve_fourier!((gx,), dst, src, (kx,))

# In-place version
deconvolve_fourier!(gs, src, ks) = deconvolve_fourier!(gs, src, src, ks)
