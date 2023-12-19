using ThreadsX: ThreadsX

abstract type AbstractBlockData end

# Dummy type used when blocking has been disabled in the NUFFT plan.
struct NullBlockData <: AbstractBlockData end
with_blocking(::NullBlockData) = false
sort_points!(::NullBlockData, xp) = nothing

struct BlockData{
        T, N,
        Tr,  # = real(T)
        Buffers <: AbstractVector{<:AbstractArray{T, N}},
        Indices <: CartesianIndices{N},
    } <: AbstractBlockData
    block_dims  :: Dims{N}        # size of each block (in number of elements)
    block_sizes :: NTuple{N, Tr}  # size of each block (in units of length)
    buffers :: Buffers        # length = nthreads
    blocks_per_thread :: Vector{Int}  # maps a set of blocks i_start:i_end to a thread (length = nthreads + 1)
    indices :: Indices    # index associated to each block (length = num_blocks)
    buffers_for_indices :: Vector{NTuple{N, Vector{Int}}}  # maps values of current buffer to indices in global array (length = nthreads())
    cumulative_npoints_per_block :: Vector{Int}    # cumulative sum of number of points in each block (length = 1 + num_blocks, initial value is 0)
    blockidx  :: Vector{Int}  # linear index of block associated to each point (length = Np)
    pointperm :: Vector{Int}  # index permutation for sorting points according to their block (length = Np)
end

function BlockData(::Type{T}, block_dims::Dims{D}, Ñs::Dims{D}, ::HalfSupport{M}) where {T, D, M}
    Nt = Threads.nthreads()
    # Nt = ifelse(Nt == 1, zero(Nt), Nt)  # this disables blocking if running on single thread
    dims = block_dims .+ 2M  # include padding for values outside of block
    Tr = real(T)
    block_sizes = map(Ñs, block_dims) do N, B
        @inline
        Δx = Tr(2π) / N  # grid step
        B * Δx
    end
    buffers = map(_ -> Array{T}(undef, dims), 1:Nt)
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
    )
end

# Blocking is considered to be disabled if there are no allocated buffers.
with_blocking(bd::BlockData) = !isempty(bd.buffers)

function sort_points!(bd::BlockData, xp::AbstractVector)
    with_blocking(bd) || return nothing
    (; indices, cumulative_npoints_per_block, blockidx, pointperm, block_sizes,) = bd
    fill!(cumulative_npoints_per_block, 0)
    to_linear_index = LinearIndices(axes(indices))  # maps Cartesian to linear index of a block
    Np = length(xp)
    resize!(blockidx, Np)
    resize!(pointperm, Np)

    @inbounds for (i, x⃗) ∈ pairs(xp)
        # Get index of block where point x⃗ is located.
        is = map(x⃗, block_sizes) do x, Δx  # we assume x⃗ is already in [0, 2π)
            # @assert 0 ≤ x < 2π
            1 + floor(Int, x / Δx)
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

    if Threads.nthreads() == 1
        # This is the same as sortperm! but seems to be faster.
        sort!(pointperm; by = i -> @inbounds(blockidx[i]), alg = QuickSort)
        # sortperm!(pointperm, blockidx; alg = QuickSort)
    else
        ThreadsX.sort!(pointperm; by = i -> @inbounds(blockidx[i]), alg = ThreadsX.QuickSort())
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
