using StaticArrays: MVector

# Interpolate onto a single point
@kernel function interpolate_to_point_naive_kernel!(
        vp::NTuple{C},
        @Const(points::NTuple{D}),
        @Const(us::NTuple{C}),
        @Const(pointperm),
        @Const(Δxs::NTuple{D}),           # grid step in each direction (oversampled grid)
        evaluate::NTuple{D, <:Function},  # can't be marked Const for some reason
        to_indices::NTuple{D, <:Function},
    ) where {C, D}
    i = @index(Global, Linear)

    j = if pointperm === nothing
        i
    else
        @inbounds pointperm[i]
    end

    x⃗ = map(xs -> @inbounds(xs[j]), points)

    # Determine grid dimensions.
    # Unlike in spreading, here `us` can be made of arrays of complex numbers, because we
    # don't perform atomic operations. This is why the code is simpler here.
    Ns = size(first(us))  # grid dimensions

    # Evaluate 1D kernels.
    gs_eval = map((f, x) -> f(x), evaluate, x⃗)

    # Determine indices to load from `u` arrays.
    indvals = ntuple(Val(D)) do n
        @inbounds begin
            gdata = gs_eval[n]
            vals = gdata.values .* Δxs[n]
            f = to_indices[n]
            f(gdata.i, Ns[n]) => vals
        end
    end

    v⃗ = interpolate_from_arrays_gpu(us, indvals)

    for n ∈ eachindex(vp, v⃗)
        @inbounds vp[n][j] = v⃗[n]
    end

    nothing
end

function interpolate!(
        backend::GPU,
        bd::Union{BlockDataGPU, NullBlockData},
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

    pointperm = get_pointperm(bd)                  # nothing in case of NullBlockData
    sort_points = get_sort_points(bd)::StaticBool  # False in the case of NullBlockData

    if pointperm !== nothing
        @assert eachindex(pointperm) == eachindex(x⃗s)
    end

    if sort_points === True()
        vp_sorted = map(similar, vp_all)  # allocate temporary arrays for sorted non-uniform data
        pointperm_ = nothing  # we don't need permutations in interpolation kernel (all accesses to non-uniform data will be contiguous)
    else
        vp_sorted = vp_all
        pointperm_ = pointperm
    end

    # We use dynamically sized kernels to avoid recompilation, since number of points may
    # change from one call to another.
    ndrange = size(x⃗s)  # iterate through points
    workgroupsize = default_workgroupsize(backend, ndrange)
    kernel! = interpolate_to_point_naive_kernel!(backend, workgroupsize)
    kernel!(vp_sorted, xs_comp, us, pointperm_, Δxs, evaluate, to_indices; ndrange)

    if sort_points === True()
        kernel_perm! = interp_permute_kernel!(backend, workgroupsize)
        kernel_perm!(vp_all, vp_sorted, pointperm; ndrange)
        foreach(KA.unsafe_free!, vp_sorted)  # manually deallocate temporary arrays
    end

    vp_all
end

@inline function interpolate_from_arrays_gpu(
        us::NTuple{C, AbstractArray{T, D}},
        indvals::NTuple{D, <:Pair},
    ) where {T, C, D}
    if @generated
        gprod_init = Symbol(:gprod_, D + 1)  # the name of this variable is important!
        Tr = real(T)
        quote
            inds_mapping = map(first, indvals)
            vals = map(last, indvals)
            inds = map(eachindex, vals)
            vs = zero(MVector{$C, $T})
            $gprod_init = one($Tr)
            @nloops(
                $D, i,
                d -> inds[d],  # for i_d ∈ inds[d]
                d -> begin
                    @inbounds gprod_d = gprod_{d + 1} * vals[d][i_d]  # add factor for dimension d
                    @inbounds j_d = inds_mapping[d][i_d]
                end,
                begin
                    # gprod_1 contains the product vals[1][i_1] * vals[2][i_2] * ...
                    js = @ntuple $D j
                    for n ∈ 1:$C
                        @inbounds vs[n] += gprod_1 * us[n][js...]
                    end
                end
            )
            Tuple(vs)
        end
    else
        # Note: the trick of splitting the first dimension from the other ones really helps with
        # performance on the GPU.
        vs = zero(MVector{C, T})
        inds_mapping = map(first, indvals)
        vals = map(last, indvals)
        inds = map(eachindex, inds_mapping)
        inds_first, inds_tail = first(inds), Base.tail(inds)
        vals_first, vals_tail = first(vals), Base.tail(vals)
        imap_first, imap_tail = first(inds_mapping), Base.tail(inds_mapping)
        @inbounds for J_tail ∈ CartesianIndices(inds_tail)
            js_tail = Tuple(J_tail)
            is_tail = map(inbounds_getindex, imap_tail, js_tail)
            gs_tail = map(inbounds_getindex, vals_tail, js_tail)
            gprod_tail = prod(gs_tail)
            for j ∈ inds_first
                i = imap_first[j]
                gprod = gprod_tail * vals_first[j]
                for n ∈ eachindex(vs)
                    vs[n] += gprod * us[n][i, is_tail...]
                end
            end
        end
        Tuple(vs)
    end
end

# This applies the *inverse* permutation by switching i ↔ j indices (opposite of spread_permute_kernel!).
@kernel function interp_permute_kernel!(vp::NTuple{N}, @Const(vp_in::NTuple{N}), @Const(perm::AbstractVector)) where {N}
    i = @index(Global, Linear)
    j = @inbounds perm[i]
    for n ∈ 1:N
        @inbounds vp[n][j] = vp_in[n][i]
    end
    nothing
end
