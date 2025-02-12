# GPU implementation
struct BlockDataGPU{
        N, I <: Integer, T <: AbstractFloat,
        IndexVector <: AbstractVector{I},
        SortPoints <: StaticBool,
        Np,
    } <: AbstractBlockData
    method :: Symbol     # method used for spreading and interpolation; options: (1) :global_memory (2) :shared_memory
    Δxs :: NTuple{N, T}  # grid step; the size of a block in units of length is block_dims .* Δxs.
    nblocks_per_dir  :: NTuple{N, I}  # number of blocks in each direction
    block_dims       :: NTuple{N, I}  # maximum dimensions of a block (excluding ghost cells)
    cumulative_npoints_per_block :: IndexVector    # cumulative sum of number of points in each block (length = num_blocks + 1, first value is 0)
    blockidx      :: IndexVector  # linear index of block associated to each point (length = Np)
    pointperm     :: IndexVector
    sort_points   :: SortPoints
    batch_size :: Val{Np}  # how many non-uniform points per batch in SM spreading?
    function BlockDataGPU(
            method::Symbol,
            Δxs::NTuple{N, T},
            nblocks_per_dir::NTuple{N, I},
            block_dims::NTuple{N, I},
            npoints_per_block::V, sort::S;
            batch_size::Val{Np},
        ) where {N, I, T, V, S, Np}
        Np::Integer
        method ∈ (:global_memory, :shared_memory) || throw(ArgumentError("expected gpu_method ∈ (:global_memory, :shared_memory)"))
        blockidx = similar(npoints_per_block, 0)
        pointperm = similar(npoints_per_block, 0)
        new{N, I, T, V, S, Np}(
            method, Δxs, nblocks_per_dir, block_dims, npoints_per_block, blockidx, pointperm, sort,
            batch_size,
        )
    end
end

gpu_method(bd::BlockDataGPU) = bd.method
with_blocking(::BlockDataGPU) = true
get_batch_size(bd::BlockDataGPU) = get_batch_size(bd.batch_size)
get_batch_size(::Val{Np}) where {Np} = Np

function BlockDataGPU(
        ::Type{Z},
        backend::KA.Backend, block_dims::Dims{D}, Ñs::Dims{D}, h::HalfSupport{M},
        sort_points::StaticBool;
        method::Symbol,
        batch_size::Val = Val(DEFAULT_GPU_BATCH_SIZE),  # batch size (in number of non-uniform points) used in spreading if gpu_method == :shared_memory
    ) where {Z <: Number, D, M}
    T = real(Z)  # in case Z is complex
    if method === :shared_memory
        # Override input block size. We try to maximise the use of shared memory.
        # Show warning if the determined block size is too small.
        block_dims, Np = block_dims_gpu_shmem(backend, Z, Ñs, h, batch_size; warn = true)
        Np_in = get_batch_size(batch_size)
        @assert Np_in == DEFAULT_GPU_BATCH_SIZE || Np_in ≤ Np  # the actual Np may be larger than the input one (to maximise shared memory usage)
    else
        # This is just for type stability: the type of `batch_size` below should be a
        # compile-time constant. Note: the returned value of Np might be negative, but we
        # don't really care because we don't use it. With `warn = false` we ensure that an
        # error is not thrown.
        _, Np = block_dims_gpu_shmem(backend, Z, Ñs, h, batch_size; warn = false)
    end
    IntType = Int  # Int32 doesn't seem to be faster
    nblocks_per_dir = IntType.(map(cld, Ñs, block_dims))  # basically equal to ceil(Ñ / block_dim)
    L = T(2) * π  # domain period
    Δxs = map(N -> L / N, Ñs)  # grid step (oversampled grid)
    cumulative_npoints_per_block = KA.allocate(backend, IntType, prod(nblocks_per_dir) + 1)
    BlockDataGPU(
        method, Δxs, nblocks_per_dir, IntType.(block_dims), cumulative_npoints_per_block, sort_points;
        batch_size = Val(Np),
    )
end

get_pointperm(bd::BlockDataGPU) = bd.pointperm

function set_points_impl!(
        backend::GPU, bd::BlockDataGPU, points::StructVector, xp, timer;
        transform::F = identity, synchronise,
    ) where {F <: Function}
    (;
        Δxs, cumulative_npoints_per_block, nblocks_per_dir, block_dims,
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
        maybe_synchronise(backend, synchronise)
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
            blockidx, cumulative_npoints_per_block, points_comp, xp, Δxs,
            block_dims, nblocks_per_dir, sort_points, transform;
            ndrange,
        )
        maybe_synchronise(backend, synchronise)
    end

    @timeit timer "(2) Cumulative sum" begin
        # Note: the Julia docs state that this can fail if the accumulation is done in
        # place. With CUDA, this doesn't seem to be a problem, but we could allocate a
        # separate array if it becomes an issue.
        cumsum!(cumulative_npoints_per_block, cumulative_npoints_per_block)
        maybe_synchronise(backend, synchronise)
    end

    # Compute permutation needed to sort points according to their block.
    @timeit timer "(3) Sort" let
        local kernel! = sortperm_kernel!(backend, workgroupsize)
        kernel!(
            pointperm, cumulative_npoints_per_block, blockidx, xp, Δxs,
            block_dims, nblocks_per_dir, transform;
            ndrange,
        )
        maybe_synchronise(backend, synchronise)
    end

    # `pointperm` now contains the permutation needed to sort points
    if sort_points === True()
        # TODO: combine this with Sort step?
        @timeit timer "(4) Permute points" let
            local kernel! = permute_kernel!(backend, workgroupsize)
            kernel!(points_comp, xp, pointperm, transform; ndrange)
            maybe_synchronise(backend, synchronise)
        end
    end

    nothing
end

# Get index of block where x⃗ is located (assumed to be in [0, 2π)).
@inline function block_index(
        x⃗::NTuple{N,T}, Δxs::NTuple{N,T}, block_dims::NTuple{N,Integer}, nblocks_per_dir::NTuple{N},
    ) where {N, T <: AbstractFloat}
    IntType = eltype(block_dims)
    is = ntuple(Val(N)) do d
        @inline
        # Note: we could directly use the block size Δx_block = Δx * block_dims, but this
        # may lead to inconsistency with interpolation and spreading kernels (leading to
        # errors) due to numerical accuracy issues. So we first obtain the index on the
        # Δx grid, then we translate that to a block index by dividing by the block
        # dimensions (= how many Δx's in a single block).
        i, r = Kernels.point_to_cell(x⃗[d], Δxs[d])  # index of grid cell where point is located (i ≥ 1)
        cld(IntType(i), block_dims[d])  # index of block where point is located
    end
    @inbounds IntType(LinearIndices(nblocks_per_dir)[is...])
end

@kernel function assign_blocks_kernel!(
        blockidx::AbstractVector{<:Integer},
        cumulative_npoints_per_block::AbstractVector{IntType},
        points::NTuple,
        @Const(xp),
        @Const(Δxs),
        @Const(block_dims::NTuple),
        @Const(nblocks_per_dir::NTuple),
        @Const(sort_points),
        @Const(transform::F),
    ) where {IntType, F}
    I::IntType = @index(Global, Linear)
    x⃗ = unsafe_get_point_as_tuple(typeof(Δxs), xp, I)
    y⃗ = to_unit_cell_gpu(transform(x⃗)) :: NTuple
    n = block_index(y⃗, Δxs, block_dims, nblocks_per_dir)::IntType

    # Note: here index_within_block is the value *after* incrementing (≥ 1).
    index_within_block = @inbounds (Atomix.@atomic cumulative_npoints_per_block[n + 1] += one(IntType))::IntType
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
        @Const(Δxs),
        @Const(block_dims),
        @Const(nblocks_per_dir),
        @Const(transform::F),
    ) where {F}
    I = @index(Global, Linear)
    x⃗ = unsafe_get_point_as_tuple(typeof(Δxs), xp, I)
    y⃗ = to_unit_cell_gpu(transform(x⃗)) :: NTuple
    n = block_index(y⃗, Δxs, block_dims, nblocks_per_dir)
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
