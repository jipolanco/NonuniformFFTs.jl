# Interpolate onto a single point
@kernel function interpolate_to_point_naive_kernel!(
        vp::NTuple{C},
        @Const(points::NTuple{D}),
        @Const(us::NTuple{C}),
        @Const(Δxs::NTuple{D}),           # grid step in each direction (oversampled grid)
        evaluate::NTuple{D, <:Function},  # can't be marked Const for some reason
        to_indices::NTuple{D, <:Function},
    ) where {C, D}
    i = @index(Global, Linear)
    x⃗ = map(xs -> @inbounds(xs[i]), points)

    # Determine grid dimensions.
    # Unlike in spreading, here `us` can be made of arrays of complex numbers, because we
    # don't perform atomic operations. This is why the code is simpler here.
    Ns = size(first(us))  # grid dimensions

    # Evaluate 1D kernels.
    gs_eval = map((f, x) -> f(x), evaluate, x⃗)

    # Determine indices to load from `u` arrays.
    inds = map(to_indices, gs_eval, Ns) do f, gdata, N
        f(gdata.i, N)
    end

    vals = map(gs_eval, Δxs) do geval, Δx
        geval.values .* Δx
    end

    # The @inline seems to help with performance (slightly, but consistently).
    v⃗ = @inline interpolate_from_arrays(us, inds, vals)

    for (dst, v) ∈ zip(vp, v⃗)
        @inbounds dst[i] = v
    end

    nothing
end

function interpolate!(
        backend::GPU,
        gs,
        vp_all::NTuple{C, AbstractVector},
        us::NTuple{C, AbstractArray},
        x⃗s::AbstractVector,
    ) where {C}
    # Note: the dimensions of arrays have already been checked via check_nufft_nonuniform_data.
    Base.require_one_based_indexing(x⃗s)  # this is to make sure that all indices match
    foreach(Base.require_one_based_indexing, vp_all)

    evaluate = map(Kernels.evaluate_kernel_func, gs)   # kernel evaluation functions
    to_indices = map(Kernels.kernel_indices_func, gs)  # functions returning spreading indices
    xs_comp = StructArrays.components(x⃗s)
    Δxs = map(Kernels.gridstep, gs)

    # TODO: use dynamically sized kernel? (to avoid recompilation, since number of points may change from one call to another)
    ndrange = size(x⃗s)  # iterate through points
    workgroupsize = default_workgroupsize(backend, ndrange)
    kernel! = interpolate_to_point_naive_kernel!(backend, workgroupsize, ndrange)
    kernel!(vp_all, xs_comp, us, Δxs, evaluate, to_indices)

    vp_all
end
