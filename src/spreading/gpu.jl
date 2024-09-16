@kernel function spread_from_points_naive_kernel!(
        us::NTuple{C}, @Const(points::NTuple{D}), @Const(vp::NTuple{C}),
        evaluate::NTuple{D, <:Function},  # can't be marked Const for some reason
        to_indices::NTuple{D, <:Function},
    ) where {C, D}
    # Spread from a single point
    i = @index(Global, Linear)
    x⃗ = map(xs -> @inbounds(xs[i]), points)
    v⃗ = map(v -> @inbounds(v[i]), vp)

    # Evaluate 1D kernels.
    gs_eval = map((f, x) -> f(x), evaluate, x⃗)

    # Determine indices to write in `u` arrays.
    Ns = size(first(us))
    inds = map(to_indices, gs_eval, Ns) do f, gdata, N
        f(gdata.i, N)
    end

    vals = map(g -> g.values, gs_eval)

    spread_onto_arrays!(us, inds, vals, v⃗; atomic = Val(true))

    nothing
end

# GPU implementation.
# We assume all arrays are already on the GPU.
function spread_from_points!(
        backend::GPU,
        gs,
        us_all::NTuple{C, AbstractGPUArray},
        x⃗s::StructVector,
        vp_all::NTuple{C, AbstractGPUVector},
    ) where {C}
    # Note: the dimensions of arrays have already been checked via check_nufft_nonuniform_data.
    Base.require_one_based_indexing(x⃗s)  # this is to make sure that all indices match
    foreach(Base.require_one_based_indexing, vp_all)
    evaluate = map(Kernels.evaluate_kernel_func, gs)   # kernel evaluation functions
    to_indices = map(Kernels.kernel_indices_func, gs)  # functions returning spreading indices
    xs_comp = StructArrays.components(x⃗s)
    ndrange = size(x⃗s)  # iterate through points
    workgroupsize = default_workgroupsize(backend, ndrange)
    kernel! = spread_from_points_naive_kernel!(backend, workgroupsize, ndrange)
    kernel!(us_all, xs_comp, vp_all, evaluate, to_indices)
end
