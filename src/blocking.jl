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
    buffers :: Buffers
    indices :: Indices
    cumulative_npoints_per_block :: Vector{Int}  # cumulative sum of number of points in each block (length = 1 + num_blocks, initial value is 0)
    blockidx  :: Vector{Int}  # linear index of block associated to each point (length = Np)
    pointperm :: Vector{Int}  # index permutation for sorting points according to their block (length = Np)
end

function BlockData(::Type{T}, block_dims::Dims{D}, Ñs::Dims{D}, ::HalfSupport{M}) where {T, D, M}
    Nt = Threads.nthreads()
    Nt = ifelse(Nt == 1, zero(Nt), Nt)  # this disables blocking if running on single thread
    dims = block_dims .+ 2M  # include padding for values outside of block
    Tr = real(T)
    block_sizes = map(Ñs, block_dims) do N, B
        @inline
        Δx = Tr(2π) / N  # grid step
        B * Δx
    end
    buffers = map(_ -> Array{T}(undef, dims), 1:Nt)
    indices = map(Ñs, block_dims) do N, B
        range(0, N - 1; step = B)
    end
    cumulative_npoints_per_block = Vector{Int}(undef, prod(block_dims) + 1)
    blockidx = Int[]
    pointperm = Int[]
    BlockData(block_dims, block_sizes, buffers, CartesianIndices(indices), cumulative_npoints_per_block, blockidx, pointperm)
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
