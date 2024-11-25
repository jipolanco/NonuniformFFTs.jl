abstract type AbstractBlockData end

get_block_dims(bd::AbstractBlockData) = bd.block_dims
get_sort_points(bd::AbstractBlockData) = bd.sort_points

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

# Resize vector trying to avoid copy when N is larger than the original length.
# In other words, we completely discard the original contents of x, which is not the
# original behaviour of resize!. This can save us some device-to-device copies.
function resize_no_copy!(x, N)
    resize!(x, 0)
    resize!(x, N)
    x
end

include("no_blocking.jl")
include("cpu.jl")
include("gpu.jl")
