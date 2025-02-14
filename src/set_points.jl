"""
    set_points!(p::PlanNUFFT, points)

Set non-uniform points before executing a NUFFT.

In one dimension, `points` is simply a vector of real values (the non-uniform locations).

In multiple dimensions, `points` may be passed as:

- a tuple of vectors `(xs::AbstractVector, ys::AbstractVector, …)` (**should be preferred**);
- a vector `[x⃗₁, x⃗₂, x⃗₃, x⃗₄, …]` of tuples or static vectors (typically `SVector`s from the
  StaticArrays.jl package);
- a matrix of size `(d, Np)` where `d` is the spatial dimension and `Np` the number of non-uniform points.

The first format should be preferred if one wants to avoid extra allocations.

The points are allowed to be outside of the periodic cell ``[0, 2π)^d``, in which case they
will be "folded" to that domain.
"""
function set_points! end

function set_points!(p::PlanNUFFT{T, N}, xp::NTuple{N, AbstractVector}; kwargs...) where {T, N}
    (; points, synchronise,) = p
    timer = get_timer_nowarn(p)
    @timeit timer "Set points" set_points_impl!(p.backend, p.blocks, points, xp, timer; synchronise, kwargs...)
    p
end

# 1D case
function set_points!(p::PlanNUFFT{T, 1}, xp::AbstractVector{<:Real}; kwargs...) where {T}
    set_points!(p, (xp,); kwargs...)
end

# Here the element type of `xp` can be either an NTuple{N, <:Real}, an SVector{N, <:Real},
# or anything else which has length `N`.
function set_points!(p::PlanNUFFT, xp::AbstractVector; kwargs...)
    N = ndims(p)
    type_length(eltype(xp)) == N || throw(DimensionMismatch(lazy"expected $N-dimensional points"))
    xp_tup = ntuple(d -> getindex.(xp, d), Val(N))  # creates copy!
    set_points!(p, xp_tup; kwargs...)
end

# Kept for backwards compatibility
set_points!(p::PlanNUFFT, xp::StructVector; kwargs...) = set_points(p, StructArrays.components(xp); kwargs...)

# Matrix as input. This version will also create a copy to switch to (xs, ys, ...) format.
# TODO: support this format only for NFFTPlan? (AbstractNFFTs interface)
function set_points!(p::PlanNUFFT, xp::AbstractMatrix{T}; kwargs...) where {T}
    N = ndims(p)
    size(xp, 1) == N || throw(DimensionMismatch(lazy"expected input matrix to have dimensions ($N, Np)"))
    if N == 1
        xp_vec = vec(xp)
    else
        xp_vec = ntuple(d -> xp[d, :], Val(N))
    end
    set_points!(p, xp_vec; kwargs...)
end
