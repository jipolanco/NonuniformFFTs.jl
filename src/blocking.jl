struct BlockData{
        T, N,
        Buffers <: AbstractVector{<:AbstractArray{T, N}},
        Indices <: CartesianIndices{N},
    }
    buffers :: Buffers
    indices :: Indices
end

function BlockData(::Type{T}, block_dims::Dims{D}, Ñs::Dims{D}, ::HalfSupport{M}) where {T, D, M}
    Nt = Threads.nthreads()
    Nt = ifelse(Nt == 1, zero(Nt), Nt)  # this disables blocking if running on single thread
    dims = block_dims .+ 2M  # include padding for values outside of block
    buffers = map(_ -> Array{T}(undef, dims), 1:Nt)
    inds = map(Ñs, block_dims) do N, B
        range(0, N - 1; step = B)
    end
    BlockData(buffers, CartesianIndices(inds))
end
