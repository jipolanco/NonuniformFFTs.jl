abstract type AbstractBlockData end

# Dummy type used when blocking has been disabled in the NUFFT plan.
struct NullBlockData <: AbstractBlockData end
with_blocking(::NullBlockData) = false

# For now these are only used in the GPU implementation
get_pointperm(::NullBlockData) = nothing
get_sort_points(::NullBlockData) = False()

@kernel function copy_points_unblocked_kernel!(@Const(transform::F), points::NTuple, @Const(xp)) where {F}
    I = @index(Global, Linear)
    @inbounds x⃗ = xp[I]
    for n ∈ eachindex(x⃗)
        @inbounds points[n][I] = to_unit_cell(transform(x⃗[n]))
    end
    nothing
end

# Here the element type of `xp` can be either an NTuple{N, <:Real}, an SVector{N, <:Real},
# or anything else which has length `N`.
function set_points!(backend, ::NullBlockData, points::StructVector, xp, timer; transform::F = identity) where {F <: Function}
    length(points) == length(xp) || resize_no_copy!(points, length(xp))
    KA.synchronize(backend)
    Base.require_one_based_indexing(points)
    @timeit timer "(1) Copy + fold" begin
        # NOTE: we explicitly iterate through StructVector components because CUDA.jl
        # currently fails when implicitly writing to a StructArray (compilation fails,
        # tested on Julia 1.11-rc3 and CUDA.jl v5.4.3).
        points_comp = StructArrays.components(points)
        kernel! = copy_points_unblocked_kernel!(backend)
        kernel!(transform, points_comp, xp; ndrange = size(xp))
        KA.synchronize(backend)  # mostly to get accurate timings
    end
    nothing
end

# "Folds" location onto unit cell [0, 2π]ᵈ.
to_unit_cell(x⃗::Tuple) = map(to_unit_cell, x⃗)

function to_unit_cell(x::Real)
    L = oftype(x, 2π)
    while x < 0
        x += L
    end
    while x ≥ L
        x -= L
    end
    x
end

# This is a bit faster on GPUs, probably because it's guaranteed to be branchless.
@inline to_unit_cell_gpu(x⃗::Tuple) = map(to_unit_cell_gpu, x⃗)
@inline to_unit_cell_gpu(x::Real) = mod2pi(x)

type_length(::Type{T}) where {T} = length(T)  # usually for SVector
type_length(::Type{<:NTuple{N}}) where {N} = N

# ================================================================================ #

# GPU implementation
struct BlockDataGPU{
        N, I <: Integer, T <: AbstractFloat,
        IndexVector <: AbstractVector{I},
        SortPoints <: StaticBool,
    } <: AbstractBlockData
    nblocks_per_dir  :: NTuple{N, I}  # number of blocks in each direction
    block_sizes      :: NTuple{N, T}  # size of each block (in units of length)
    cumulative_npoints_per_block :: IndexVector    # cumulative sum of number of points in each block (length = num_blocks + 1, first value is 0)
    blockidx      :: IndexVector  # linear index of block associated to each point (length = Np)
    pointperm     :: IndexVector
    sort_points   :: SortPoints
    function BlockDataGPU(
            nblocks_per_dir::NTuple{N, I},
            block_sizes::NTuple{N, T},
            npoints_per_block::V, sort::S,
        ) where {N, I, T, V, S}
        blockidx = similar(npoints_per_block, 0)
        pointperm = similar(npoints_per_block, 0)
        new{N, I, T, V, S}(
            nblocks_per_dir, block_sizes, npoints_per_block, blockidx, pointperm, sort,
        )
    end
end

with_blocking(::BlockDataGPU) = true

function BlockDataGPU(
        ::Type{Z},
        backend::KA.Backend, block_dims::Dims{D}, Ñs::Dims{D}, sort_points,
    ) where {Z <: Number, D}
    T = real(Z)  # in case Z is complex
    nblocks_per_dir = map(cld, Ñs, block_dims)  # basically equal to ceil(Ñ / block_dim) --> effective block size is ≤ block_dims
    L = T(2) * π
    block_sizes = map(nblocks -> L / nblocks, nblocks_per_dir)
    cumulative_npoints_per_block = KA.allocate(backend, Int, prod(nblocks_per_dir) + 1)
    BlockDataGPU(nblocks_per_dir, block_sizes, cumulative_npoints_per_block, sort_points)
end

get_pointperm(bd::BlockDataGPU) = bd.pointperm
get_sort_points(bd::BlockDataGPU) = bd.sort_points

function set_points!(
        backend::GPU, bd::BlockDataGPU, points::StructVector, xp, timer;
        transform::F = identity,
    ) where {F <: Function}
    (;
        cumulative_npoints_per_block, nblocks_per_dir, block_sizes,
        blockidx, pointperm, sort_points,
    ) = bd

    npoints_max = typemax(eltype(bd.pointperm))
    length(points) ≤ npoints_max || error(lazy"number of points exceeds maximum allowed: $npoints_max")

    @assert length(cumulative_npoints_per_block) == prod(nblocks_per_dir) + 1
    Np = length(xp)

    @timeit timer "(0) Init arrays" begin
        resize_no_copy!(blockidx, Np)
        resize_no_copy!(pointperm, Np)
        resize_no_copy!(points, Np)
        fill!(cumulative_npoints_per_block, 0)
        KA.synchronize(backend)
    end

    # We avoid passing a StructVector to the kernel, so we pass `points` as a tuple of
    # vectors. The kernel might fail if `xp` is also a StructVector, which is not imposed
    # nor disallowed.
    points_comp = StructArrays.components(points)

    ndrange = size(points)
    workgroupsize = default_workgroupsize(backend, ndrange)
    @timeit timer "(1) Assign blocks" let
        local kernel! = assign_blocks_kernel!(backend, workgroupsize)
        kernel!(
            blockidx, cumulative_npoints_per_block, points_comp, xp,
            block_sizes, nblocks_per_dir, sort_points, transform;
            ndrange,
        )
        KA.synchronize(backend)
    end

    @timeit timer "(2) Cumulative sum" begin
        # Note: the Julia docs state that this can fail if the accumulation is done in
        # place. With CUDA, this doesn't seem to be a problem, but we could allocate a
        # separate array if it becomes an issue.
        cumsum!(cumulative_npoints_per_block, cumulative_npoints_per_block)
        KA.synchronize(backend)
    end

    # Compute permutation needed to sort points according to their block.
    @timeit timer "(3) Sort" let
        local kernel! = sortperm_kernel!(backend, workgroupsize)
        kernel!(
            pointperm, cumulative_npoints_per_block, blockidx, xp,
            block_sizes, nblocks_per_dir, transform;
            ndrange,
        )
        KA.synchronize(backend)
    end

    # `pointperm` now contains the permutation needed to sort points
    if sort_points === True()
        # TODO: combine this with Sort step?
        @timeit timer "(4) Permute points" let
            local kernel! = permute_kernel!(backend, workgroupsize)
            kernel!(points_comp, xp, pointperm, transform; ndrange)
            KA.synchronize(backend)
        end
    end

    nothing
end

# Get index of block where x⃗ is located (assumed to be in [0, 2π)).
@inline function block_index(
        x⃗::NTuple{N,T}, block_sizes::NTuple{N,T}, nblocks_per_dir::NTuple{N},
    ) where {N, T <: AbstractFloat}
    is = map(Kernels.point_to_cell, x⃗, block_sizes)
    @inbounds LinearIndices(nblocks_per_dir)[is...]
end

@kernel function assign_blocks_kernel!(
        blockidx::AbstractVector{<:Integer},
        cumulative_npoints_per_block::AbstractVector{<:Integer},
        points::NTuple,
        @Const(xp),
        @Const(block_sizes::NTuple),
        @Const(nblocks_per_dir::NTuple),
        @Const(sort_points),
        @Const(transform::F),
    ) where {F}
    I = @index(Global, Linear)
    @inbounds x⃗ = xp[I]
    y⃗ = to_unit_cell_gpu(transform(Tuple(x⃗))) :: NTuple
    n = block_index(y⃗, block_sizes, nblocks_per_dir)

    # Note: here index_within_block is the value *after* incrementing (≥ 1).
    S = eltype(cumulative_npoints_per_block)
    index_within_block = @inbounds (Atomix.@atomic cumulative_npoints_per_block[n + 1] += one(S))::S
    @inbounds blockidx[I] = index_within_block

    # If points need to be sorted, then we fill `points` some time later (in permute_kernel!).
    if sort_points === False()
        for n ∈ eachindex(x⃗)
            @inbounds points[n][I] = y⃗[n]
        end
    end

    nothing
end

@kernel function sortperm_kernel!(
        pointperm::AbstractVector,
        @Const(cumulative_npoints_per_block),
        @Const(blockidx),
        @Const(xp),
        @Const(block_sizes),
        @Const(nblocks_per_dir),
        @Const(transform::F),
    ) where {F}
    I = @index(Global, Linear)
    @inbounds x⃗ = xp[I]
    y⃗ = to_unit_cell_gpu(transform(Tuple(x⃗))) :: NTuple
    n = block_index(y⃗, block_sizes, nblocks_per_dir)
    @inbounds J = cumulative_npoints_per_block[n] + blockidx[I]
    @inbounds pointperm[J] = I
    nothing
end

@kernel function permute_kernel!(
        points::NTuple,
        @Const(xp::AbstractVector),
        @Const(perm::AbstractVector{<:Integer}),
        @Const(transform::F),
    ) where {F}
    i = @index(Global, Linear)
    j = @inbounds perm[i]
    x⃗ = @inbounds xp[j]
    y⃗ = to_unit_cell_gpu(transform(Tuple(x⃗))) :: NTuple
    for n ∈ eachindex(x⃗)
        @inbounds points[n][i] = y⃗[n]
    end
    nothing
end

# Resize vector trying to avoid copy when N is larger than the original length.
# In other words, we completely discard the original contents of x, which is not the
# original behaviour of resize!. This can save us some device-to-device copies.
function resize_no_copy!(x, N)
    resize!(x, 0)
    resize!(x, N)
    x
end

# ================================================================================ #

# CPU only
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
    cumulative_npoints_per_block :: Vector{Int}    # cumulative sum of number of points in each block (length = 1 + num_blocks, first value is 0)
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
    dims = block_dims .+ 2M  # include padding for values outside of block (TODO: include padding in original block_dims? requires minimal block_size in each direction)
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
    cumulative_npoints_per_block = Vector{Int}(undef, nblocks + 1)
    blockidx = Int[]
    pointperm = Int[]
    blocks_per_thread = zeros(Int, Nt + 1)
    BlockData(
        block_dims, block_sizes, buffers, blocks_per_thread, indices,
        cumulative_npoints_per_block, blockidx, pointperm,
        sort_points,
    )
end

function set_points!(backend::CPU, bd::BlockData, points::StructVector, xp, timer; transform::F = identity) where {F <: Function}
    # This technically never happens, but we might use it as a way to disable blocking.
    isempty(bd.buffers) && return set_points!(backend, NullBlockData(), points, xp, timer; transform)

    (; indices, cumulative_npoints_per_block, blockidx, pointperm, block_sizes,) = bd
    N = type_length(eltype(xp))  # = number of dimensions
    @assert N == length(block_sizes)

    @timeit timer "(0) Init arrays" begin
        to_linear_index = LinearIndices(axes(indices))  # maps Cartesian to linear index of a block
        Np = length(xp)
        resize_no_copy!(blockidx, Np)
        resize_no_copy!(pointperm, Np)
        resize_no_copy!(points, Np)
        fill!(cumulative_npoints_per_block, 0)
        KA.synchronize(backend)
    end

    @timeit timer "(1) Assign blocks" @inbounds for (i, x⃗) ∈ pairs(xp)
        # Get index of block where point x⃗ is located.
        y⃗ = to_unit_cell(transform(NTuple{N}(x⃗)))  # converts `x⃗` to Tuple if it's an SVector
        is = map(Kernels.point_to_cell, y⃗, block_sizes)
        if bd.sort_points === False()
            points[i] = y⃗  # copy folded point (doesn't need to be sorted)
        end
        # checkbounds(indices, CartesianIndex(is))
        n = to_linear_index[is...]  # linear index of block
        index_within_block = (cumulative_npoints_per_block[n + 1] += 1)  # ≥ 1
        blockidx[i] = index_within_block
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

    @timeit timer "(2) Sort" begin
        # Note: we don't use threading since it seems to be much slower.
        # This is very likely due to false sharing (https://en.wikipedia.org/wiki/False_sharing),
        # since all threads modify the same data in "random" order.
        @inbounds for i ∈ eachindex(xp)
            # We recompute the block index associated to this point.
            x⃗ = xp[i]
            y⃗ = to_unit_cell(transform(NTuple{N}(x⃗)))  # converts `x⃗` to Tuple if it's an SVector
            is = map(Kernels.point_to_cell, y⃗, block_sizes)
            n = to_linear_index[is...]  # linear index of block
            j = cumulative_npoints_per_block[n] + blockidx[i]
            pointperm[j] = i
        end
    end

    # Write sorted points into `points`.
    # (This should rather be called "permute" instead of "sort"...)
    # Note: we don't use threading since it seems to be much slower.
    # This is very likely due to false sharing (https://en.wikipedia.org/wiki/False_sharing),
    # since all threads modify the same data in "random" order.
    if bd.sort_points === True()
        @timeit timer "(3) Permute points" begin
            # TODO: combine this with Sort step?
            @inbounds for j ∈ eachindex(pointperm)
                i = pointperm[j]
                x⃗ = xp[i]
                y⃗ = to_unit_cell(transform(NTuple{N}(x⃗)))  # converts `x⃗` to Tuple if it's an SVector
                points[j] = y⃗
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
