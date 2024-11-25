# Dummy type used when blocking has been disabled in the NUFFT plan.
struct NullBlockData <: AbstractBlockData end
with_blocking(::NullBlockData) = false
get_block_dims(::NullBlockData) = nothing
get_sort_points(::NullBlockData) = False()

# For now these are only used in the GPU implementation
get_pointperm(::NullBlockData) = nothing
gpu_method(::NullBlockData) = :global_memory

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
function set_points_impl!(
        backend, ::NullBlockData, points::StructVector, xp, timer;
        synchronise, transform::F = identity
    ) where {F <: Function}
    length(points) == length(xp) || resize_no_copy!(points, length(xp))
    maybe_synchronise(backend, synchronise)
    Base.require_one_based_indexing(points)
    @timeit timer "(1) Copy + fold" begin
        # NOTE: we explicitly iterate through StructVector components because CUDA.jl
        # currently fails when implicitly writing to a StructArray (compilation fails,
        # tested on Julia 1.11-rc3 and CUDA.jl v5.4.3).
        points_comp = StructArrays.components(points)
        kernel! = copy_points_unblocked_kernel!(backend)
        kernel!(transform, points_comp, xp; ndrange = size(xp))
        maybe_synchronise(backend, synchronise)
    end
    nothing
end
