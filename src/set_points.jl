"""
    set_points!(p::PlanNUFFT, points)

Set non-uniform points before executing a NUFFT.

In one dimension, `points` is simply a vector of real values (the non-uniform locations).

In multiple dimensions, `points` may be passed as:

- a tuple of vectors `(xs::AbstractVector, ys::AbstractVector, …)`,
- a vector `[x⃗₁, x⃗₂, x⃗₃, x⃗₄, …]` of tuples or static vectors (typically `SVector`s from the
  StaticArrays.jl package).

The points are allowed to be outside of the periodic cell ``[0, 2π)^d``, in which case they
will be "folded" to that domain.
"""
function set_points! end

function set_points!(p::PlanNUFFT{T, N}, xp::NTuple{N, AbstractVector}) where {T, N}
    set_points!(p, StructVector(xp))
end

# 1D case
function set_points!(p::PlanNUFFT{T, 1}, xp::AbstractVector{<:Real}) where {T}
    set_points!(p, StructVector((xp,)))
end

# Here the element type of `xp` can be either an NTuple{N, <:Real}, an SVector{N, <:Real},
# or anything else which has length `N`.
function set_points!(p::PlanNUFFT{T, N}, xp::AbstractVector) where {T, N}
    (; points, timer,) = p
    type_length(eltype(xp)) == N || throw(DimensionMismatch(lazy"expected $N-dimensional points"))
    @timeit timer "Set points" begin
        @timeit timer "Copy/fold" begin
            resize!(points, length(xp))
            Base.require_one_based_indexing(points)
            @inbounds for (i, x) ∈ enumerate(xp)
                points[i] = to_unit_cell(NTuple{N}(x))  # converts `x` to Tuple if it's an SVector
            end
        end
        @timeit timer "Sort" begin
            sort_points!(p.blocks, points)
        end
    end
    p
end

# "Folds" location onto unit cell [0, 2π]ᵈ.
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
