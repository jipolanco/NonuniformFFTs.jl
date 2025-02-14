# Dummy type used when blocking has been disabled in the NUFFT plan.
struct NullBlockData <: AbstractBlockData end
with_blocking(::NullBlockData) = false
get_block_dims(::NullBlockData) = nothing
get_sort_points(::NullBlockData) = False()

# For now these are only used in the GPU implementation
get_pointperm(::NullBlockData) = nothing
gpu_method(::NullBlockData) = :global_memory

@kernel function copy_points_unblocked_kernel!(@Const(transform::F), points::NTuple{N}, @Const(xp::NTuple{N})) where {F, N}
    I = @index(Global, Linear)
    for n in 1:N
        @inbounds points[n][I] = to_unit_cell(transform(xp[n][I]))
    end
    nothing
end

function set_points_impl!(
        backend, point_transform_fold::F, ::NullBlockData, points_ref::Ref, timer;
        synchronise,
    ) where {F <: Function}
    xp = points_ref[]
    Np = length(xp[1])
    all(x -> length(x) == Np, xp) || throw(DimensionMismatch("input points must have the same length along all dimensions"))
    # Nothing to do!
    nothing
end
