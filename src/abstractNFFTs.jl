# AbstractNFFTs.jl interface
#
# Some current limitations of this interface:
#
# - it only works for complex non-uniform data
# - it doesn't allow simultaneous (directional) transforms (can be fixed within NonuniformFFTs.jl)

using LinearAlgebra: LinearAlgebra, Adjoint

AbstractNFFTs.size_in(p::PlanNUFFT) = size(p)  # array dimensions in uniform space
AbstractNFFTs.size_out(p::PlanNUFFT) = (length(p.points),)

# Uniform to non-uniform
function LinearAlgebra.mul!(
        vp::AbstractVector{Tc}, p::PlanNUFFT{Tc}, ûs::AbstractArray{Tc},
    ) where {Tc <: Complex}
    exec_type2!(vp, p, ûs)
    vp
end

# Non-uniform to uniform
function LinearAlgebra.mul!(
        ûs::AbstractArray{Tc},
        p::Adjoint{Tc, <:PlanNUFFT{Tc}},
        vp::AbstractVector{Tc},
    ) where {Tc <: Complex}
    exec_type1!(ûs, p, vp)
    vp
end

# Transform from AbstractNFFTs convention (x ∈ [-1/2, 1/2)) to NonuniformFFTs convention
# (x ∈ [0, 2π)). We also switch the sign convention of the Fourier transform (corresponds to
# doing x → -x).
@inline function _transform_point_convention(x::AbstractFloat)  # x ∈ [-1/2, 1/2)
    T = typeof(x)
    π_ = T(π)
    x = 2 * π_ * x  # in [-π, π)
    x = -x  # in (-π, π]
    x + π_  # in (0, 2π]  // [0, 2π) would be even better, but this works too
end

@inline _transform_point_convention(x⃗::NTuple) = map(_transform_point_convention, x⃗)

# Takes locations in [-1/2, 1/2)ᵈ, so we need to transform the point convention.
# Note: the `transform` argument is an internal of set_points! and is not documented.
# It takes an NTuple (a point location x⃗) as input.
function AbstractNFFTs.nodes!(p::PlanNUFFT, xp::AbstractMatrix{<:AbstractFloat})
    set_points!(p, xp; transform = _transform_point_convention)
end

Base.@constprop :aggressive function convert_window_function(w::Symbol)
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
        default_kernel()
    end
end

# Create plan constructor which is compatible with AbstractNFFTs (takes non-uniform points
# in a matrix as first input).
# Only complex-to-complex plans can be constructed using this interface.
# We use @constprop to hopefully avoid some type instabilities when converting some
# variables to the static domain (not sure if this helps).
Base.@constprop :aggressive function PlanNUFFT(
        xp::AbstractMatrix{Tr}, Ns::Dims;
        fftflags = FFTW.ESTIMATE, blocking = true, sortNodes = false,
        window = default_kernel(),
        kwargs...,
    ) where {Tr <: AbstractFloat}
    # Note: the NFFT.jl package uses an odd window size, w = 2m + 1.
    # Here we use an even window size w = 2m, which should result in a slightly lower
    # accuracy for the same m.
    m, σ, reltol = AbstractNFFTs.accuracyParams(; kwargs...)
    sort_points = sortNodes ? True() : False()  # this is type-unstable (unless constant propagation happens)
    block_size = blocking ? default_block_size() : nothing  # also type-unstable
    kernel = window isa AbstractKernel ? window : convert_window_function(window)
    p = PlanNUFFT(Complex{Tr}, Ns, HalfSupport(m); σ = Tr(σ), sort_points, block_size, kernel, fftw_flags = fftflags)
    AbstractNFFTs.nodes!(p, xp)
    p
end
