# CPU only
struct BlockDataCPU{
        Z, N, Nc,
        I <: Integer,
        T,  # = real(Z)
        Buffers <: AbstractVector{<:NTuple{Nc, AbstractArray{Z, N}}},
        Indices <: CartesianIndices{N},
        SortPoints <: StaticBool,
    } <: AbstractBlockData
    # The following fields are the same as in BlockDataGPU:
    Δxs :: NTuple{N, T}  # grid step; the size of a block in units of length is block_dims .* Δxs.
    nblocks_per_dir  :: NTuple{N, I}  # number of blocks in each direction
    block_dims  :: NTuple{N, I}        # size of each block (in number of elements)
    cumulative_npoints_per_block :: Vector{Int}    # cumulative sum of number of points in each block (length = 1 + num_blocks, first value is 0)
    blockidx  :: Vector{Int}  # linear index of block associated to each point (length = Np)
    pointperm :: Vector{Int}  # index permutation for sorting points according to their block (length = Np)
    sort_points :: SortPoints

    buffers :: Buffers            # length = nthreads
    blocks_per_thread :: Vector{Int}  # maps a set of blocks i_start:i_end to a thread (length = nthreads + 1)
    indices :: Indices    # index associated to each block (length = num_blocks)

    function BlockDataCPU(
            Δxs::NTuple{N, T},
            nblocks_per_dir::NTuple{N, I},
            block_dims::NTuple{N, I},
            npoints_per_block::Vector{Int},
            buffers::AbstractVector{<:NTuple{Nc, AbstractArray{Z, N}}},
            indices::Indices,
            sort::S,
        ) where {Z <: Number, Nc, N, I, T, Indices, S}
        @assert T === real(Z)
        Nt = length(buffers)
        blockidx = similar(npoints_per_block, 0)
        pointperm = similar(npoints_per_block, 0)
        blocks_per_thread = similar(blockidx, Nt + 1)
        new{Z, N, Nc, I, T, typeof(buffers), Indices, S}(
            Δxs, nblocks_per_dir, block_dims, npoints_per_block, blockidx, pointperm, sort,
            buffers, blocks_per_thread, indices,
        )
    end
end

function BlockDataCPU(
        ::Type{Z}, block_dims_in::Dims{D}, Ñs::Dims{D}, ::HalfSupport{M}, num_transforms::Val{Nc},
        sort_points::StaticBool,
    ) where {Z <: Number, D, M, Nc}
    @assert Nc > 0
    # Reduce block size if the actual dataset is too small.
    # The block size must satisfy B ≤ N - M (this is assumed in spreading/interpolation).
    block_dims = map(Ñs, block_dims_in) do N, B
        min(B, N - M)
    end
    nblocks_per_dir = map(cld, Ñs, block_dims)  # basically equal to ceil(Ñ / block_dim)
    T = real(Z)
    L = T(2) * π  # domain period
    Δxs = map(N -> L / N, Ñs)  # grid step (oversampled grid)
    cumulative_npoints_per_block = Vector{Int}(undef, prod(nblocks_per_dir) + 1)
    dims = block_dims .+ 2M  # include padding for values outside of block (TODO: include padding in original block_dims? requires minimal block_size in each direction)
    Nt = Threads.nthreads()
    # Nt = ifelse(Nt == 1, zero(Nt), Nt)  # this disables blocking if running on single thread
    buffers = map(1:Nt) do _
        ntuple(_ -> Array{Z}(undef, dims), num_transforms)  # one buffer per transform (or "component")
    end
    indices_tup = map(Ñs, block_dims) do N, B
        range(0, N - 1; step = B)
    end
    indices = CartesianIndices(indices_tup)
    BlockDataCPU(
        Δxs, nblocks_per_dir, block_dims, cumulative_npoints_per_block,
        buffers, indices,
        sort_points,
    )
end

# This is similar to assign_blocks_kernel! (GPU implementation), but using `to_unit_cell_cpu`
# instead of `to_unit_cell_gpu` (which does seem to make a difference in performance).
function assign_blocks_cpu!(
        blockidx::AbstractVector{<:Integer},
        cumulative_npoints_per_block::AbstractVector{<:Integer},
        xp::NTuple{N},
        Δxs::NTuple{N},
        block_dims::NTuple{N},
        nblocks_per_dir::NTuple{N},
        transform_fold::F,
    ) where {N, F}
    Threads.@threads :static for I ∈ eachindex(xp...)
        y⃗ = unsafe_get_point(transform_fold, xp, I)
        n = block_index(y⃗, Δxs, block_dims, nblocks_per_dir)

        # Note: here index_within_block is the value *after* incrementing (≥ 1).
        S = eltype(cumulative_npoints_per_block)
        index_within_block = @inbounds (Atomix.@atomic cumulative_npoints_per_block[n + 1] += one(S))::S
        @inbounds blockidx[I] = index_within_block
    end
    nothing
end

function sortperm_cpu!(
        pointperm::AbstractVector,
        cumulative_npoints_per_block,
        blockidx,
        xp::NTuple{N},
        Δxs::NTuple{N},
        block_dims,
        nblocks_per_dir,
        transform_fold::F,
    ) where {N, F}
    Threads.@threads :static for I ∈ eachindex(xp...)
        y⃗ = unsafe_get_point(transform_fold, xp, I)
        n = block_index(y⃗, Δxs, block_dims, nblocks_per_dir)
        @inbounds J = cumulative_npoints_per_block[n] + blockidx[I]
        @inbounds pointperm[J] = I
    end
    nothing
end

function set_points_impl!(
        backend::CPU, point_transform_fold::F, bd::BlockDataCPU, points_ref::Ref, timer;
        synchronise,
    ) where {F <: Function}
    # This technically never happens, but we might use it as a way to disable blocking.
    isempty(bd.buffers) && return set_points_impl!(backend, point_transform_fold, NullBlockData(), points_ref, timer; synchronise)

    (;
        Δxs, cumulative_npoints_per_block, nblocks_per_dir, block_dims,
        blockidx, pointperm, sort_points,
    ) = bd

    xp = points_ref[]
    N = length(xp)  # number of dimensions
    Np = length(xp[1])
    all(x -> length(x) == Np, xp) || throw(DimensionMismatch("input points must have the same length along all dimensions"))

    @assert N == length(block_dims)  # number of dimensions

    @timeit timer "(0) Init arrays" begin
        resize_no_copy!(blockidx, Np)
        resize_no_copy!(pointperm, Np)
        fill!(cumulative_npoints_per_block, 0)
    end

    @timeit timer "(1) Assign blocks" let
        assign_blocks_cpu!(
            blockidx, cumulative_npoints_per_block, xp, Δxs,
            block_dims, nblocks_per_dir, point_transform_fold,
        )
    end

    # Compute cumulative sum (we don't use cumsum! due to aliasing warning in its docs).
    @inbounds for i ∈ eachindex(IndexLinear(), cumulative_npoints_per_block)[2:end]
        cumulative_npoints_per_block[i] += cumulative_npoints_per_block[i - 1]
    end
    @assert cumulative_npoints_per_block[begin] == 0
    @assert cumulative_npoints_per_block[end] == Np

    # Determine how many blocks each thread will manage. The idea is that, if the point
    # distribution is inhomogeneous, then more threads are dedicated to areas where points
    # are concentrated, improving load balance.
    map_blocks_to_threads!(bd.blocks_per_thread, cumulative_npoints_per_block)

    @timeit timer "(2) Sort" begin
        sortperm_cpu!(
            pointperm, cumulative_npoints_per_block, blockidx, xp, Δxs,
            block_dims, nblocks_per_dir, point_transform_fold,
        )
    end

    # Write sorted points into `points`.
    # We need to allocate a new array to hold the result.
    # (This should rather be called "permute" instead of "sort"...)
    # Note: we don't use threading since it seems to be much slower.
    # This is very likely due to false sharing (https://en.wikipedia.org/wiki/False_sharing),
    # since all threads modify the same data in "random" order.
    if sort_points === True()
        @timeit timer "(3) Permute points" let
            # TODO: combine this with Sort step?
            points_ref[] = map(similar, xp)  # allocate new array
            points = points_ref[]
            @inbounds for j ∈ eachindex(pointperm)
                i = pointperm[j]
                for n in 1:N
                    points[n][j] = @inbounds xp[n][i]  # note: we could apply the transform here and not at each NUFFT
                end
            end
        end
    end

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

