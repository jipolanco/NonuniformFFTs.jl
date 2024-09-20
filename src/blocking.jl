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

type_length(::Type{T}) where {T} = length(T)  # usually for SVector
type_length(::Type{<:NTuple{N}}) where {N} = N

# ================================================================================ #

# Maximum number of bits allowed for Hilbert sort.
# Hilbert sorting divides the domain in a grid of size N^d, where `d` is the dimension and
# N = 2^b. Here `b` is the number of bits used in the algorithm.
# This means that `b = 16` corresponds to a grid of dimensions 65536 *in each direction*,
# which is far larger than anything we will ever need in practice.
# (Also note that the Hilbert "grid" is generally coarser than the actual NUFFT grid where
# values are spread and interpolated.)
const MAX_BITS_HILBERT_SORT = 16

# GPU implementation
struct BlockDataGPU{
        PermVector <: AbstractVector{<:Integer},
        SortPoints <: StaticBool,
    } <: AbstractBlockData
    pointperm     :: PermVector
    nbits_hilbert :: Int  # number of bits required in Hilbert sorting
    sort_points   :: SortPoints

    BlockDataGPU(pointperm::P, nbits, sort::S) where {P, S} =
        new{P, S}(pointperm, nbits, sort)
end

with_blocking(::BlockDataGPU) = true

# In the case of Hilbert sorting, what we call "block" corresponds to the size of the
# minimal box in the Hilbert curve algorithm.
function BlockDataGPU(backend::KA.Backend, block_dims::Dims{D}, Ñs::Dims{D}, sort_points) where {D}
    # We use a Hilbert sorting algorithm which requires the same number of blocks in each
    # direction. In the best case, blocks will have the size required by `block_dims`. If
    # that's not possible, we will prefer to use a smaller block size (so more blocks).
    # This makes sense if the input block size was tuned as proportional to some memory
    # limit (e.g. shared memory size on GPUs).
    nblocks_per_dir_wanted = map(Ñs, block_dims) do N, B
        cld(N, B)  # ≥ 1
    end
    nblocks = max(nblocks_per_dir_wanted...)  # this makes the actual block size ≤ the wanted block size
    nbits = exponent(nblocks - 1) + 1
    @assert UInt(2)^(nbits - 1) < nblocks ≤ UInt(2)^nbits  # verify that we got the right value of nbits
    nbits = min(nbits, MAX_BITS_HILBERT_SORT)  # just in case; in practice nbits < MAX_BITS_HILBERT_SORT all the time
    pointperm = KA.allocate(backend, Int, 0)   # stores permutation indices
    BlockDataGPU(pointperm, nbits, sort_points)
end

get_pointperm(bd::BlockDataGPU) = bd.pointperm
get_sort_points(bd::BlockDataGPU) = bd.sort_points

function set_points!(
        backend::GPU, bd::BlockDataGPU, points::StructVector, xp, timer;
        transform::F = identity,
    ) where {F <: Function}

    # Recursively iterate over possible values of nbits_hilbert to avoid type instability.
    # We start by nbits = 1, and increase until nbits == bd.nbits_hilbert.
    @assert bd.nbits_hilbert ≤ MAX_BITS_HILBERT_SORT
    _set_points_hilbert!(Val(1), backend, bd, points, xp, timer, transform)

    nothing
end

@inline function _set_points_hilbert!(::Val{nbits}, backend, bd, points, xp, args...) where {nbits}
    if nbits > MAX_BITS_HILBERT_SORT  # this is to avoid the compiler from exploding if it thinks the recursion is infinite
        nothing
    elseif nbits == bd.nbits_hilbert
        N = type_length(eltype(xp))  # = number of dimensions
        alg = GlobalGrayStatic(nbits, N)  # should be inferred, since nbits is statically known
        _set_points_hilbert!(alg, backend, bd, points, xp, args...)  # call method defined further below
    else
        _set_points_hilbert!(Val(nbits + 1), backend, bd, points, xp, args...)  # keep iterating
    end
end

function _set_points_hilbert!(
        sortalg::HilbertSortingAlgorithm, backend, bd::BlockDataGPU,
        points::StructVector, xp, timer, transform::F,
    ) where {F}
    (; pointperm, sort_points,) = bd

    B = nbits(sortalg)  # static value
    nblocks = 1 << B    # same number of blocks in all directions (= 2^B)

    Np = length(xp)
    resize_no_copy!(pointperm, Np)
    resize_no_copy!(points, Np)
    @assert eachindex(points) == eachindex(xp)

    # Allocate temporary array for holding Hilbert indices (manually deallocated later)
    T = eltype(sortalg)
    inds = KA.zeros(backend, T, Np)

    # We avoid passing a StructVector to the kernel, so we pass `points` as a tuple of
    # vectors. The kernel might fail if `xp` is also a StructVector, which is not imposed
    # nor disallowed.
    points_comp = StructArrays.components(points)

    ndrange = size(points)
    groupsize = default_workgroupsize(backend, ndrange)
    kernel! = hilbert_sort_kernel!(backend, groupsize, ndrange)
    @timeit timer "(1) Hilbert encoding" begin
        kernel!(inds, points_comp, xp, sortalg, nblocks, sort_points, transform)
        KA.synchronize(backend)
    end

    # Compute permutation needed to sort Hilbert indices.
    @timeit timer "(2) Sortperm" begin
        sortperm!(pointperm, inds)
        KA.synchronize(backend)
    end

    KA.unsafe_free!(inds)

    # `pointperm` now contains the permutation needed to sort points
    if sort_points === True()
        @timeit timer "(3) Permute points" let
            local kernel! = permute_kernel!(backend, groupsize, ndrange)
            kernel!(points_comp, xp, pointperm, transform)
            KA.synchronize(backend)
        end
    end

    nothing
end

# This kernel may do multiple things at once:
# - Compute Hilbert index associated to a point
# - Copy point from `xp` to `points`, after transformations and folding, if sort_points === False().
@kernel function hilbert_sort_kernel!(
        inds::AbstractVector{<:Unsigned},
        points::NTuple,
        @Const(xp),
        @Const(sortalg::HilbertSortingAlgorithm),
        @Const(nblocks::Integer),  # TODO: can we pass this as a compile-time constant? (Val)
        @Const(sort_points),
        @Const(transform::F),
    ) where {F}
    I = @index(Global, Linear)
    @inbounds x⃗ = xp[I]
    y⃗ = to_unit_cell(transform(Tuple(x⃗))) :: NTuple
    T = eltype(y⃗)
    @assert T <: AbstractFloat

    L = T(2) * π
    block_size = L / nblocks

    is = map(y⃗) do y
        i = Kernels.point_to_cell(y, block_size)
        i - one(i)  # Hilbert sorting requires zero-based indices
    end

    @inbounds inds[I] = encode_hilbert_zero(sortalg, is)  # compute Hilbert index for sorting

    # If points need to be sorted, then we fill `points` some time later (in permute_kernel!).
    if sort_points === False()
        for n ∈ eachindex(x⃗)
            @inbounds points[n][I] = y⃗[n]
        end
    end

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
    y⃗ = to_unit_cell(transform(Tuple(x⃗))) :: NTuple  # note: we perform the transform + folding twice per point...
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
    dims = block_dims .+ 2M  # include padding for values outside of block (TODO: include padding in original block_dims?)
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

function set_points!(::CPU, bd::BlockData, points::StructVector, xp, timer; transform::F = identity) where {F <: Function}
    # This technically never happens, but we might use it as a way to disable blocking.
    isempty(bd.buffers) && return set_points!(NullBlockData(), points, xp, timer)

    (; indices, cumulative_npoints_per_block, blockidx, pointperm, block_sizes,) = bd
    N = type_length(eltype(xp))  # = number of dimensions
    @assert N == length(block_sizes)
    fill!(cumulative_npoints_per_block, 0)
    to_linear_index = LinearIndices(axes(indices))  # maps Cartesian to linear index of a block
    Np = length(xp)
    resize_no_copy!(blockidx, Np)
    resize_no_copy!(pointperm, Np)
    resize_no_copy!(points, Np)

    @timeit timer "(1) Assign blocks" @inbounds for (i, x⃗) ∈ pairs(xp)
        # Get index of block where point x⃗ is located.
        y⃗ = to_unit_cell(transform(NTuple{N}(x⃗)))  # converts `x⃗` to Tuple if it's an SVector
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
                y⃗ = to_unit_cell(transform(NTuple{N}(x⃗)))  # converts `x⃗` to Tuple if it's an SVector
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
