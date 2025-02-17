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

For convenience (and backwards compatibility), one can also pass a `StructVector` from the
StructArrays.jl package, which is equivalent to the first option above.

The points are allowed to be outside of the periodic cell ``[0, 2π)^d``, in which case they
will be "folded" to that domain.

!!! note "Modifying the points arrays"

    As one may expect, this package does not modify the input points, as doing this would be surprising.
    However, to avoid allocations and copies, it keeps a reference to the point data when performing the transforms.
    This means that one should **not** modify the passed arrays between a call to `set_points!` and [`exec_type1!`](@ref)
    or [`exec_type2!`](@ref), as this can lead to wrong results or much worse.

"""
function set_points! end

function set_points!(p::PlanNUFFT{Z, N}, xp::NTuple{N, AbstractVector{T}}; kwargs...) where {Z, N, T}
    (; points_ref, synchronise,) = p
    T === real(Z) || throw(ArgumentError(lazy"input points must have the same accuracy as the created plan (got $T points for a $Z plan)"))
    P_in = typeof(xp)
    P_plan = eltype(points_ref)
    P_in === P_plan || throw(
        ArgumentError(
            lazy"""unexpected point container:
            - expected:  points::$P_plan
            - got:       points::$P_in"""
        )
    )
    points_ref[] = xp  # copy "pointer"
    timer = get_timer_nowarn(p)
    @timeit timer "Set points" set_points_impl!(
        p.backend, p.point_transform_fold, p.blocks, points_ref, timer;
        synchronise, kwargs...,
    )
    return p
end

# 1D case
function set_points!(p::PlanNUFFT{T, 1}, xp::AbstractVector{<:Real}; kwargs...) where {T}
    set_points!(p, (xp,); kwargs...)
end

# Here the element type of `xp` can be either an NTuple{N, <:Real}, an SVector{N, <:Real},
# or anything else which has length `N`.
# Note that this creates a copy!
function set_points!(p::PlanNUFFT, xp::AbstractVector; kwargs...)
    N = ndims(p)
    type_length(eltype(xp)) == N || throw(DimensionMismatch(lazy"expected $N-dimensional points"))
    T = eltype(eltype(xp))
    @assert T <: Real
    xp_tup = ntuple(Val(N)) do d
        yp = similar(xp, T, length(xp))
        yp .= getindex.(xp, d)
    end
    set_points!(p, xp_tup; kwargs...)
end

# Kept for backwards compatibility
# TODO: move to an extension?
set_points!(p::PlanNUFFT, xp::StructVector; kwargs...) = set_points(p, StructArrays.components(xp); kwargs...)

# Matrix as input. This version will also create a copy to switch to (xs, ys, ...) format.
# TODO: support this format only for NFFTPlan? (AbstractNFFTs interface)
function set_points!(p::PlanNUFFT, xp::AbstractMatrix{T}; kwargs...) where {T}
    N = ndims(p)
    size(xp, 1) == N || throw(DimensionMismatch(lazy"expected input matrix to have dimensions ($N, Np)"))
    if N == 1
        xp_tup = (vec(xp),)
    else
        xp_tup = ntuple(Val(N)) do d
            yp = similar(xp, T, size(xp, 2))
            yp .= @view xp[d, :]
        end
    end
    set_points!(p, xp_tup; kwargs...)
end
