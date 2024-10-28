# AbstractNFFTs.jl interface
#
# Some current limitations of this interface:
#
# - it only works for complex non-uniform data
# - it doesn't allow simultaneous (directional) transforms (can be fixed within NonuniformFFTs.jl)

using LinearAlgebra: LinearAlgebra, Adjoint

# This is a wrapper type allowing to define an interface which is compatible with
# AbstractNFFTs.jl. It is not exported to avoid clashes with NFFT.jl.
"""
    NonuniformFFTs.NFFTPlan(xp::AbstractMatrix{T}, dims::Dims{D}; kwargs...) -> plan

Create a NUFFT plan which is compatible with the
[AbstractNFFTs.jl](https://juliamath.github.io/NFFT.jl/stable/abstract/) interface.

The plan follows different rules from plans created by [`PlanNUFFT`](@ref).
In particular:

- points are assumed to be in ``[-1/2, 1/2)`` instead of ``[0, 2π)``;
- the opposite Fourier sign convention is used (e.g. ``e^{-i k x_j}`` becomes ``e^{+2π i k x_j}``);
- uniform data is in increasing order by default, with frequencies ``k = -N/2, …, -1, 0,
  1, …, N/2-1``, as opposed to preserving the order used by FFTW (which starts at ``k = 0``);
- it only supports complex non-uniform data.

This constructor requires passing the non-uniform locations `xp` as the first argument.
These should be given as a matrix of dimensions `(D, Np)`, where `D` is the spatial
dimension and `Np` the number of non-uniform points.

The second argument is simply the size `(N₁, N₂, …)` of the uniform data arrays.

This constructor creates a plan which assumes complex-valued non-uniform data.
For real-valued data, the [`PlanNUFFT`](@ref) constructor should be used instead.

# Compatibility with NFFT.jl

Most of the [parameters](https://juliamath.github.io/NFFT.jl/stable/overview/#Parameters)
supported by the NFFT.jl package are also supported by this constructor.
The currently supported parameters are `reltol`, `m`, `σ`, `window`, `blocking`, `sortNodes` and `fftflags`.

Moreover, unlike [`PlanNUFFT`](@ref), this constructor sets `fftshift = true` by default (but
can be overridden) so that the uniform data ordering is the same as in NFFT.jl.

!!! warning "Type instability"

    Explicitly passing some of these parameters may result in type-unstable code, since the
    exact type of the returned plan cannot be inferred.
    This is because, in NonuniformFFTs.jl, parameters such as the kernel size (`m`) or the
    convolution window (`window`) are included in the plan type (they are compile-time constants).
    
# GPU usage

To create a GPU-compatible plan, simply pass the locations `xp` as a GPU array (e.g. a `CuArray` in CUDA).
Unlike [`PlanNUFFT`](@ref), the `backend` argument is not needed here and will be simply ignored.
"""
struct NFFTPlan{
        T <: AbstractFloat, N, Plan <: PlanNUFFT{Complex{T}, N, 1},
    } <: AbstractNFFTPlan{T, N, 1}
    p :: Plan
end

function Base.show(io::IO, p::NFFTPlan{T, N}) where {T, N}
    print(io, "NonuniformFFTs.NFFTPlan{$T, $N} wrapping a PlanNUFFT:\n")
    print(io, p.p)
end

AbstractNFFTs.size_in(p::NFFTPlan) = size(p.p)  # array dimensions in uniform space
AbstractNFFTs.size_out(p::NFFTPlan) = (length(p.p.points),)

# Uniform to non-uniform
function LinearAlgebra.mul!(
        vp::AbstractVector{Complex{T}}, p::NFFTPlan{T}, ûs::AbstractArray{Complex{T}},
    ) where {T}
    exec_type2!(vp, p.p, ûs)
    vp
end

# Non-uniform to uniform
function LinearAlgebra.mul!(
        ûs::AbstractArray{Complex{T}, N},
        p::Adjoint{Complex{T}, <:NFFTPlan{T, N}},
        vp::AbstractVector{Complex{T}},
    ) where {T, N}
    exec_type1!(ûs, parent(p).p, vp)
    vp
end

# Transform from AbstractNFFTs convention (x ∈ [-1/2, 1/2)) to NonuniformFFTs convention
# (x ∈ [0, 2π)). We also switch the sign convention of the Fourier transform (corresponds to
# doing x → -x).
@inline function _transform_point_convention(x::AbstractFloat)  # x ∈ [-1/2, 1/2)
    T = typeof(x)
    twopi = 2 * T(π)
    x = twopi * x  # in [-π, π)
    x = -x  # in (-π, π]
    ifelse(x < 0, x + twopi, x)  # in [0, 2π)
end

@inline _transform_point_convention(x⃗::NTuple) = map(_transform_point_convention, x⃗)

# Takes locations in [-1/2, 1/2)ᵈ, so we need to transform the point convention.
# Note: the `transform` argument is an internal of set_points! and is not documented.
# It takes either an NTuple (a point location x⃗) or a scalar (coordinate xᵢ) as input.
function AbstractNFFTs.nodes!(p::NFFTPlan, xp::AbstractMatrix{T}) where {T <: AbstractFloat}
    set_points!(p.p, xp; transform = _transform_point_convention)
end

# This is to avoid ambiguity issues, since AbstractNFFTs defines nodes! for Matrix instead
# of AbstractMatrix.
function AbstractNFFTs.nodes!(p::NFFTPlan{T}, xp::Matrix{T}) where {T <: AbstractFloat}
    invoke(AbstractNFFTs.nodes!, Tuple{NFFTPlan, AbstractMatrix{T}}, p, xp)
end

Base.@constprop :aggressive function convert_window_function(w::Symbol, backend)
    if w === :gauss
        GaussianKernel()
    elseif w === :spline
        BSplineKernel()
    elseif w === :kaiser_bessel_rev
        # We seem to use opposite conventions compared to NFFT.jl, as their
        # :kaiser_bessel_rev corresponds to our KaiserBesselKernel (while our
        # BackwardsKaiserBesselKernel corresponds to :kaiser_bessel).
        KaiserBesselKernel()
    elseif w === :kaiser_bessel
        BackwardsKaiserBesselKernel()
    else
        default_kernel(backend)
    end
end

# Create plan constructor which is compatible with AbstractNFFTs (takes non-uniform points
# in a matrix as first input).
# Only complex-to-complex plans can be constructed using this interface.
# WARN: this constructor is type unstable (the return type is not inferred) when explicitly
# passed certain keyword arguments.
# We use @constprop to hopefully avoid some type instabilities, but that doesn't seem to
# fully help here.
# TODO: support all keyword arguments of the standard constructor
Base.@constprop :aggressive function NFFTPlan(
        xp::AbstractMatrix{T}, Ns::Dims;
        fftflags = FFTW.ESTIMATE, blocking = true, sortNodes = false,
        window = default_kernel(KA.get_backend(xp)),
        fftshift = true,  # for compatibility with NFFT.jl
        synchronise = false,
        kwargs...,
    ) where {T <: AbstractFloat}
    # Note: the NFFT.jl package uses an odd window size, w = 2m + 1.
    # Here we use an even window size w = 2m, which should result in a slightly lower
    # accuracy for the same m.
    m, σ, reltol = AbstractNFFTs.accuracyParams(; kwargs...)
    backend = KA.get_backend(xp)  # e.g. use GPU backend if xp is a GPU array
    sort_points = sortNodes ? True() : False()  # this is type-unstable (unless constant propagation happens)
    block_size = blocking ? default_block_size(Ns, backend) : nothing  # also type-unstable
    kernel = window isa AbstractKernel ? window : convert_window_function(window, backend)
    p = PlanNUFFT(
        Complex{T}, Ns, HalfSupport(m);
        backend, σ = T(σ), sort_points, fftshift, block_size,
        kernel, fftw_flags = fftflags, synchronise,
    )
    pp = NFFTPlan(p)
    AbstractNFFTs.nodes!(pp, xp)
    pp
end
