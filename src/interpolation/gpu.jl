using StaticArrays: MVector

# Interpolate onto a single point
@kernel function interpolate_to_point_naive_kernel!(
        vp::NTuple{C},
        @Const(gs::NTuple{D}),
        @Const(points::NTuple{D}),
        @Const(us::NTuple{C}),
        @Const(pointperm),
        @Const(Δxs::NTuple{D}),           # grid step in each direction (oversampled grid)
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

    indvals = ntuple(Val(D)) do n
        @inline
        g = gs[n]
        gdata = Kernels.evaluate_kernel(g, x⃗[n])
        vals = gdata.values    # kernel values
        M = Kernels.half_support(gs[n])
        i₀ = gdata.i - M  # active region is (i₀ + 1):(i₀ + 2M) (up to periodic wrapping)
        i₀ = ifelse(i₀ < 0, i₀ + Ns[n], i₀)  # make sure i₀ ≥ 0
        i₀ => vals
    end

    v⃗ = interpolate_from_arrays_gpu(us, indvals, Ns, Δxs)

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
    kernel!(vp_sorted, gs, xs_comp, us, pointperm_, Δxs; ndrange)

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
        Ns::Dims{D},
        Δxs::NTuple{D, Tr},
    ) where {T, Tr, C, D}
    if @generated
        gprod_init = Symbol(:gprod_, D + 1)  # the name of this variable is important!
        quote
            @assert Tr === real(T)
            inds_start = map(first, indvals)  # start of active region in input array
            vals = map(last, indvals)    # evaluated kernel values in each direction
            inds = map(eachindex, vals)  # = (1:L, 1:L, ...) where L = 2M is the kernel width
            vs = zero(MVector{$C, $T})   # interpolated value (output)
            $gprod_init = prod(Δxs)  # product of kernel values (initially Δx[1] * Δx[2] * ...)
            @nloops(
                $D, i,
                d -> inds[d],  # for i_d ∈ 1:L
                d -> begin
                    @inbounds gprod_d = gprod_{d + 1} * vals[d][i_d]  # add factor for dimension d
                    @inbounds j_d = inds_start[d] + i_d
                    @inbounds j_d = ifelse(j_d > Ns[d], j_d - Ns[d], j_d)  # periodic wrapping
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
        # Fallback implementation in case the @generated version above doesn't work.
        # Note: the trick of splitting the first dimension from the other ones really helps with
        # performance on the GPU.
        @assert Tr === real(T)
        vs = zero(MVector{C, T})
        inds_start = map(first, indvals)
        vals = map(last, indvals)
        inds = map(eachindex, vals)
        inds_first, inds_tail = first(inds), Base.tail(inds)
        vals_first, vals_tail = first(vals), Base.tail(vals)
        istart_first, istart_tail = first(inds_start), Base.tail(inds_start)
        N, Ns_tail = first(Ns), Base.tail(Ns)
        gprod_base = prod(Δxs)  # this factor is needed in interpolation only
        @inbounds for I_tail ∈ CartesianIndices(inds_tail)
            is_tail = Tuple(I_tail)
            gs_tail = map(inbounds_getindex, vals_tail, is_tail)
            gprod_tail = gprod_base * prod(gs_tail)
            js_tail = map(istart_tail, is_tail, Ns_tail) do j₀, i, Nloc
                # Determine input index in the current dimension.
                @inline
                j = j₀ + i
                ifelse(j > Nloc, j - Nloc, j)  # periodic wrapping
            end
            for i ∈ inds_first
                j = istart_first + i
                j = ifelse(j > N, j - N, j)  # periodic wrapping
                js = (j, js_tail...)
                gprod = gprod_tail * vals_first[i]
                for n ∈ eachindex(vs)
                    vs[n] += gprod * us[n][js...]
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
