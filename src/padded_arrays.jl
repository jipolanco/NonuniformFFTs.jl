"""
    PaddedArray{M, T, N} <: AbstractArray{T, N}

Pads a vector with `M` "ghost" entries on each side, along each direction.

Can be useful for dealing with periodic boundary conditions.

---

    PaddedArray{M}(data::AbstractArray)

Interpret input array as a padded array.

Note that the input array is not modified. Instead, its `M` first and `M` last
entries along each direction are considered as "ghost" entries.

In other words, the "logical" dimensions of the resulting `PaddedArray` are
`size(v) = size(data) .- 2M`.
Along a given direction of size `N`, indexing functions like `axes` return the range `1:N`
(or an equivalent).
However, the array can in reality be indexed (and modified) over the range `(1 - M):(N + M)`.
"""
struct PaddedArray{M, T, N, A <: AbstractArray{T, N}} <: AbstractArray{T, N}
    data :: A
    function PaddedArray{M}(data::AbstractArray) where {M}
        A = typeof(data)
        T = eltype(A)
        N = ndims(A)
        Base.require_one_based_indexing(data)
        new{M, T, N, A}(data)
    end
end

npad(::Type{<:PaddedArray{M}}) where {M} = M
npad(v::AbstractArray) = npad(typeof(v))

# General case of N-D arrays: we can't use linear indexing due to the padding in all
# directions.
Base.IndexStyle(::Type{<:PaddedArray}) = IndexCartesian()

# Case of 1D arrays (vectors). Returns IndexLinear() if `A <: Vector`.
Base.IndexStyle(::Type{<:PaddedArray{M, T, 1, A}}) where {M, T, A} = IndexStyle(A)

Base.parent(v::PaddedArray) = v.data
Base.size(v::PaddedArray) = ntuple(i -> size(parent(v), i) - 2 * npad(v), Val(ndims(v)))

# Return a view to the parent array.
function Base.view(v::PaddedArray{M, T, N}, inds::Vararg{Any, N}) where {M, T, N}
    inds_parent = map(inds, axes(v)) do is, js
        to_parent_indices(M, is, js)
    end
    view(parent(v), inds_parent...)
end

to_parent_indices(M, i::Integer, js) = M + i
to_parent_indices(M, is::AbstractVector, js) = M .+ is  # this includes ranges
to_parent_indices(M, ::Colon, js) = M .+ js  # select all values excluding ghost cells

function Base.copyto!(w::PaddedArray{M}, v::PaddedArray{M}) where {M}
    length(w.data) == length(v.data) || throw(DimensionMismatch("arrays have different sizes"))
    copyto!(w.data, v.data)
    w
end

# Equality and isapprox must also include ghost cells!
Base.:(==)(w::PaddedArray{M}, v::PaddedArray{M}) where {M} = w.data == v.data

Base.isapprox(w::PaddedArray{M}, v::PaddedArray{M}; kwargs...) where {M} =
    isapprox(w.data, v.data; kwargs...)

function Base.similar(v::PaddedArray{M, T, N}, ::Type{S}, dims::Dims{N}) where {S, M, T, N}
    PaddedArray{M}(similar(v.data, S, dims .+ 2M))
end

Base.@propagate_inbounds function Base.getindex(v::PaddedArray{M, T, N}, I::Vararg{Int, N}) where {M, T, N}
    J = ntuple(n -> I[n] + M, Val(N))
    parent(v)[J...]
end

Base.@propagate_inbounds function Base.setindex!(
        v::PaddedArray{M, T, N},
        val,
        I::Vararg{Int, N},
    ) where {M, T, N}
    J = I .+ M
    parent(v)[J...] = val
end

Base.checkbounds(::Type{Bool}, v::PaddedArray, I...) = _checkbounds(v, I...)

_checkbounds(v::PaddedArray, I::CartesianIndex) = _checkbounds(v, Tuple(I)...)

function _checkbounds(v::PaddedArray, Is...)
    M = npad(v)
    N = ndims(v)
    P = length(Is)
    @assert P ≥ N
    Is_head = ntuple(n -> Is[n], Val(N))
    Is_tail = ntuple(n -> Is[N + n], Val(P - N))  # possible additional indices which should be all 1
    for (inds, i) ∈ zip(axes(v), Is_head)
        _checkbounds(Val(M), inds, i) || return false
    end
    all(isone, Is_tail)
end

_checkbounds(::Val{M}, inds::AbstractUnitRange, i::Integer) where {M} =
    first(inds) - M ≤ i ≤ last(inds) + M

_checkbounds(::Val{M}, inds::AbstractUnitRange, I::AbstractUnitRange) where {M} =
    first(inds) - M ≤ first(I) && last(I) ≤ last(inds) + M

## ================================================================================ ##
## Periodic padding.
## ================================================================================ ##

# Copy values from main (non-ghost) region to ghost cells.
@inline function copy_to_ghost!(v::PaddedArray)
    # Copy ghost cells over each dimension separately.
    N = ndims(v)
    _copy_to_ghost!(Val(N), v)
end

# Copy values over dimension d.
@inline function _copy_to_ghost!(::Val{d}, v::PaddedArray) where {d}
    data = parent(v)
    N = ndims(data)
    inds_before = CartesianIndices(ntuple(i -> axes(data, i), Val(d - 1)))
    inds_after = CartesianIndices(ntuple(i -> axes(data, d + i), Val(N - d)))
    ibegin = firstindex(data, d)
    iend = lastindex(data, d)
    M = npad(v)
    @inbounds for J ∈ inds_after, I ∈ inds_before
        for δ ∈ 1:M  # copy each ghost cell layer
            data[I, iend - M + δ, J] = data[I, ibegin - 1 + M + δ, J]  # copy value to ghost cell
            data[I, ibegin - 1 + δ, J] = data[I, iend - 2M + δ, J]
        end
    end
    _copy_to_ghost!(Val(d - 1), v)
end

# Stop when we have copied over all dimensions.
@inline _copy_to_ghost!(::Val{0}, v::PaddedArray) = v

## ================================================================================ ##

# Add values from ghost cells to the main region.
@inline function add_from_ghost!(v::PaddedArray)
    N = ndims(v)
    _add_from_ghost!(Val(N), v)
end

# Add values over dimension d.
@inline function _add_from_ghost!(::Val{d}, v::PaddedArray) where {d}
    data = parent(v)
    N = ndims(data)
    inds_before = CartesianIndices(ntuple(i -> axes(data, i), Val(d - 1)))
    inds_after = CartesianIndices(ntuple(i -> axes(data, d + i), Val(N - d)))
    ibegin = firstindex(data, d)
    iend = lastindex(data, d)
    M = npad(v)
    @inbounds for J ∈ inds_after, I ∈ inds_before
        for δ ∈ 1:M
            data[I, ibegin - 1 + M + δ, J] += data[I, iend - M + δ, J]  # add value from ghost cell
            data[I, iend - 2M + δ, J] += data[I, ibegin - 1 + δ, J]
        end
    end
    _add_from_ghost!(Val(d - 1), v)
end

@inline _add_from_ghost!(::Val{0}, v::PaddedArray) = v
