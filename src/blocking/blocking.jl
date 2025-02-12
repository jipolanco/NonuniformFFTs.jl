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

# This is faster on GPUs, probably because it's guaranteed to be branchless.
@inline to_unit_cell_gpu(x⃗::Tuple) = map(to_unit_cell_gpu, x⃗)

@inline function to_unit_cell_gpu(x::Real)
    twopi = oftype(x, 2) * π
    r = rem(x, twopi)  # note: rem(x, y) translates to fmodf/fmod on CUDA (see https://github.com/JuliaGPU/CUDA.jl/blob/4067511b2b472be9fb30164d1ed23caa354c1fcb/src/device/intrinsics/math.jl#L370)
    # This is adapted from Julia's mod implementation (also based on rem), but trying to
    # avoid branches (not sure it improves much).
    r = ifelse(iszero(r), copysign(r, twopi), r)  # replaces -0.0 -> +0.0
    ifelse(r < 0, twopi + r, r)
end

type_length(::Type{T}) where {T} = length(T)  # usually for SVector
type_length(::Type{<:NTuple{N}}) where {N} = N

# Get point from vector of points, converting it to a tuple of the wanted type T.
# The first argument is only used to determine the output type.
# The "unsafe" is because we apply @inbounds.
function unsafe_get_point_as_tuple(
        ::Type{NTuple{D, T}},
        xp::AbstractVector,
        i::Integer,
    ) where {D, T <: AbstractFloat}
    ntuple(Val(D)) do d
        @inbounds T(xp[i][d])
    end
end

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
