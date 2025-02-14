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
        backend, ::NullBlockData, points::NTuple{N}, xp::NTuple{N}, timer;
        synchronise, transform::F = identity,
    ) where {F <: Function, N}
    Np = length(xp[1])
    all(x -> length(x) == Np, xp) || throw(DimensionMismatch("input points must have the same length along all dimensions"))
    length(points[1]) == Np || resize_no_copy!(points, Np)
    maybe_synchronise(backend, synchronise)
    foreach(Base.require_one_based_indexing, points)
    foreach(Base.require_one_based_indexing, xp)
    @timeit timer "(1) Copy + fold" begin
        kernel! = copy_points_unblocked_kernel!(backend)
        kernel!(transform, points, xp; ndrange = size(xp[1]))
        maybe_synchronise(backend, synchronise)
    end
    nothing
end
