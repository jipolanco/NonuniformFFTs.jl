abstract type AbstractBlockData end

# Dummy type used when blocking has been disabled in the NUFFT plan.
struct NullBlockData <: AbstractBlockData end
with_blocking(::NullBlockData) = false

# Here the element type of `xp` can be either an NTuple{N, <:Real}, an SVector{N, <:Real},
# or anything else which has length `N`.
function set_points!(::NullBlockData, points, xp, timer)
    resize!(points, length(xp))
    Base.require_one_based_indexing(points)
    N = type_length(eltype(xp))
    @timeit timer "(1) Copy + fold" begin
        @inbounds for (i, x) ∈ enumerate(xp)
            points[i] = to_unit_cell(NTuple{N}(x))  # converts `x` to Tuple if it's an SVector
        end
    end
    nothing
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

# ================================================================================ #

struct BlockData{
        T, N, Nc,
        Tr,  # = real(T)
        Buffers <: AbstractVector{<:NTuple{Nc, AbstractArray{T, N}}},
        Indices <: CartesianIndices{N},
        SortPoints <: StaticBool,
    } <: AbstractBlockData
    block_dims  :: Dims{N}        # size of each block (in number of elements)
    block_sizes :: NTuple{N, Tr}  # size of each block (in units of length)
    buffers :: Buffers            # length = nthreads
    blocks_per_thread :: Vector{Int}  # maps a set of blocks i_start:i_end to a thread (length = nthreads + 1)
    indices :: Indices    # index associated to each block (length = num_blocks)
    buffers_for_indices :: Vector{NTuple{N, Vector{Int}}}  # maps values of current buffer to indices in global array (length = nthreads())
    cumulative_npoints_per_block :: Vector{Int}    # cumulative sum of number of points in each block (length = 1 + num_blocks, initial value is 0)
    blockidx  :: Vector{Int}  # linear index of block associated to each point (length = Np)
    pointperm :: Vector{Int}  # index permutation for sorting points according to their block (length = Np)
    sort_points :: SortPoints
end

function BlockData(
        ::Type{T}, block_dims::Dims{D}, Ñs::Dims{D}, ::HalfSupport{M}, num_transforms::Val{Nc},
        sort_points::StaticBool,
    ) where {T, D, M, Nc}
    @assert Nc > 0
    Nt = Threads.nthreads()
    # Nt = ifelse(Nt == 1, zero(Nt), Nt)  # this disables blocking if running on single thread
    # Reduce block size if the total grid size is not sufficiently large in a given
    # direction. This maximum block size is assumed in spreading and interpolation.
    block_dims = map(Ñs, block_dims) do N, B
        @assert N - M > 0
        min(B, N ÷ 2, N - M)
    end
    dims = block_dims .+ 2M  # include padding for values outside of block
    Tr = real(T)
    block_sizes = map(Ñs, block_dims) do N, B
        @inline
        Δx = Tr(2π) / N  # grid step
        B * Δx
    end
    buffers = map(1:Nt) do _
        ntuple(_ -> Array{T}(undef, dims), num_transforms)  # one buffer per transform (or "component")
    end
    indices_tup = map(Ñs, block_dims) do N, B
        range(0, N - 1; step = B)
    end
    indices = CartesianIndices(indices_tup)
    nblocks = length(indices)  # total number of blocks
    buffers_for_indices = Vector{NTuple{D, Vector{Int}}}(undef, Nt)
    for i ∈ eachindex(buffers_for_indices)
        buffers_for_indices[i] = map(N -> Vector{Int}(undef, N), dims)
    end
    cumulative_npoints_per_block = Vector{Int}(undef, nblocks + 1)
    blockidx = Int[]
    pointperm = Int[]
    blocks_per_thread = zeros(Int, Nt + 1)
    BlockData(
        block_dims, block_sizes, buffers, blocks_per_thread, indices, buffers_for_indices,
        cumulative_npoints_per_block, blockidx, pointperm,
        sort_points,
    )
end

# Blocking is considered to be disabled if there are no allocated buffers.
with_blocking(bd::BlockData) = !isempty(bd.buffers)

function set_points!(bd::BlockData, points, xp, timer)
    with_blocking(bd) || return set_points!(NullBlockData(), points, xp, timer)

    (; indices, cumulative_npoints_per_block, blockidx, pointperm, block_sizes,) = bd
    N = type_length(eltype(xp))  # = number of dimensions
    @assert N == length(block_sizes)
    fill!(cumulative_npoints_per_block, 0)
    to_linear_index = LinearIndices(axes(indices))  # maps Cartesian to linear index of a block
    Np = length(xp)
    resize!(blockidx, Np)
    resize!(pointperm, Np)
    resize!(points, Np)

    @timeit timer "(1) Assign blocks" @inbounds for (i, x⃗) ∈ pairs(xp)
        # Get index of block where point x⃗ is located.
        y⃗ = to_unit_cell(NTuple{N}(x⃗))  # converts `x⃗` to Tuple if it's an SVector
        is = map(Kernels.point_to_cell, y⃗, block_sizes)
        if bd.sort_points === False()
            points[i] = y⃗  # copy folded point (doesn't need to be sorted)
        end
        # checkbounds(indices, CartesianIndex(is))
        n = to_linear_index[is...]  # linear index of block
        cumulative_npoints_per_block[n + 1] += 1
        pointperm[i] = i
        blockidx[i] = n
    end

    # Compute cumulative sum (we don't use cumsum! due to aliasing warning in its docs).
    for i ∈ eachindex(IndexLinear(), cumulative_npoints_per_block)[2:end]
        cumulative_npoints_per_block[i] += cumulative_npoints_per_block[i - 1]
    end
    @assert cumulative_npoints_per_block[begin] == 0
    @assert cumulative_npoints_per_block[end] == Np

    # Determine how many blocks each thread will manage. The idea is that, if the point
    # distribution is inhomogeneous, then more threads are dedicated to areas where points
    # are concentrated, improving load balance.
    map_blocks_to_threads!(bd.blocks_per_thread, cumulative_npoints_per_block)

    @timeit timer "(2) Sortperm" begin
        quicksort_perm!(pointperm, blockidx)
    end

    # Write sorted points into `points`.
    # (This should rather be called "permute" instead of "sort"...)
    # Note: we don't use threading since it seems to be much slower.
    # This is very likely due to false sharing (https://en.wikipedia.org/wiki/False_sharing),
    # since all threads modify the same data in "random" order.
    if bd.sort_points === True()
        @timeit timer "(3) Permute points" begin
            @inbounds for i ∈ eachindex(pointperm)
                j = pointperm[i]
                x⃗ = xp[j]
                y⃗ = to_unit_cell(NTuple{N}(x⃗))  # converts `x⃗` to Tuple if it's an SVector
                points[i] = y⃗
            end
        end
    end

    # Verification
    # for i ∈ eachindex(cumulative_npoints_per_block)[begin:end - 1]
    #     a = cumulative_npoints_per_block[i] + 1
    #     b = cumulative_npoints_per_block[i + 1]
    #     for j ∈ a:b
    #         @assert blockidx[pointperm[j]] == i
    #     end
    # end

    nothing
end

function map_blocks_to_threads!(blocks_per_thread, cumulative_npoints_per_block)
    Np = last(cumulative_npoints_per_block)  # total number of points
    Nt = length(blocks_per_thread) - 1       # number of threads
    Np_per_thread = Np / Nt  # target number of points per thread
    blocks_per_thread[begin] = 0
    @assert cumulative_npoints_per_block[begin] == 0
    n = firstindex(cumulative_npoints_per_block) - 1
    nblocks = length(cumulative_npoints_per_block) - 1
    Base.require_one_based_indexing(cumulative_npoints_per_block)
    for i ∈ 1:Nt
        npoints_in_current_thread = 0
        stop = false
        while npoints_in_current_thread < Np_per_thread
            n += 1
            if n > nblocks
                stop = true
                break
            end
            npoints_in_block = cumulative_npoints_per_block[n + 1] - cumulative_npoints_per_block[n]
            npoints_in_current_thread += npoints_in_block
        end
        if stop
            blocks_per_thread[begin + i] = nblocks  # this thread ends at the last block (inclusive)
            for j ∈ (i + 1):Nt
                blocks_per_thread[begin + j] = nblocks  # this thread does no work (starts and ends at the last block)
            end
            break
        else
            blocks_per_thread[begin + i] = n  # this thread ends at block `n` (inclusive)
        end
    end
    blocks_per_thread[end] = nblocks  # make sure the last block is included
    blocks_per_thread
end
