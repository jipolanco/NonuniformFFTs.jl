using StaticArrays: MVector

# Interpolate onto a single point
@kernel function interpolate_to_point_naive_kernel!(
        vp::NTuple{C},
        @Const(gs::NTuple{D}),
        @Const(evalmode::EvaluationMode),
        @Const(points::NTuple{D}),
        @Const(us::NTuple{C}),
        @Const(pointperm),
        @Const(prefactor::Real),    # = volume of a grid cell = prod(Δxs)
    ) where {C, D}
    i = @index(Global, Linear)

    j = if pointperm === nothing
        i
    else
        @inbounds pointperm[i]
    end

    # Determine grid dimensions.
    # Unlike in spreading, here `us` can be made of arrays of complex numbers, because we
    # don't perform atomic operations. This is why the code is simpler here.
    Ns = size(first(us))  # grid dimensions

    indvals = get_inds_vals_gpu(gs, evalmode, points, Ns, j)

    v⃗ = interpolate_from_arrays_gpu(us, indvals, Ns, prefactor)

    for n ∈ eachindex(vp, v⃗)
        @inbounds vp[n][j] = v⃗[n]
    end

    nothing
end

function interpolate!(
        backend::GPU,
        bd::Union{BlockDataGPU, NullBlockData},
        gs::NTuple{D},
        evalmode::EvaluationMode,
        vp_all::NTuple{C, AbstractVector},
        us::NTuple{C, AbstractArray},
        x⃗s::AbstractVector,
    ) where {C, D}
    # Note: the dimensions of arrays have already been checked via check_nufft_nonuniform_data.
    Base.require_one_based_indexing(x⃗s)  # this is to make sure that all indices match
    foreach(Base.require_one_based_indexing, vp_all)

    xs_comp = StructArrays.components(x⃗s)
    Δxs = map(Kernels.gridstep, gs)
    prefactor = prod(Δxs)  # interpolations need to be multiplied by the volume of a grid cell

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
    ndrange_points = size(x⃗s)  # iterate through points
    groupsize_points = default_workgroupsize(backend, ndrange_points)

    method = gpu_method(bd)

    if method === :global_memory
        let ndrange = ndrange_points, groupsize = groupsize_points
            kernel! = interpolate_to_point_naive_kernel!(backend, groupsize)
            kernel!(vp_sorted, gs, evalmode, xs_comp, us, pointperm_, prefactor; ndrange)
        end
    elseif method === :shared_memory
        @assert bd isa BlockDataGPU
        Z = eltype(us[1])
        M = Kernels.half_support(gs[1])
        @assert all(g -> Kernels.half_support(g) === M, gs)  # check that they're all equal
        block_dims_val, batch_size_actual = block_dims_gpu_shmem(backend, Z, size(us[1]), HalfSupport(M), bd.batch_size)  # this is usually a compile-time constant...
        @assert Val(batch_size_actual) == bd.batch_size
        block_dims = Val(block_dims_val)  # ...which means this doesn't require a dynamic dispatch
        @assert block_dims_val === bd.block_dims
        let ngroups = bd.nblocks_per_dir  # this is the required number of workgroups (number of blocks in CUDA)
            block_dims_padded = @. block_dims_val + 2M - 1  # dimensions of shared memory array
            shmem_size = block_dims_padded
            groupsize = 64
            ndrange = gpu_shmem_ndrange_from_groupsize(groupsize, ngroups)
            kernel! = interpolate_to_points_shmem_kernel!(backend, groupsize, ndrange)
            kernel!(
                vp_sorted, gs, evalmode, xs_comp, us, pointperm_, bd.cumulative_npoints_per_block,
                prefactor,
                block_dims, Val(shmem_size),
            )
        end
    end

    if sort_points === True()
        let ndrange = ndrange_points, groupsize = groupsize_points
            kernel_perm! = interp_permute_kernel!(backend, groupsize)
            kernel_perm!(vp_all, vp_sorted, pointperm; ndrange)
            foreach(KA.unsafe_free!, vp_sorted)  # manually deallocate temporary arrays
        end
    end

    vp_all
end

@inline function interpolate_from_arrays_gpu(
        us::NTuple{C, AbstractArray{T, D}},
        indvals::NTuple{D, <:Pair},
        Ns::Dims{D},
        prefactor::Tr,
    ) where {T, Tr, C, D}
    if @generated
        gprod_init = Symbol(:gprod_, D + 1)  # the name of this variable is important!
        quote
            @assert Tr === real(T)
            inds_start = map(first, indvals)  # start of active region in input array
            vals = map(last, indvals)    # evaluated kernel values in each direction
            inds = map(eachindex, vals)  # = (1:L, 1:L, ...) where L = 2M is the kernel width
            vs = zero(MVector{$C, $T})   # interpolated value (output)
            $gprod_init = prefactor  # product of kernel values (initially Δx[1] * Δx[2] * ...)
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
        gprod_base = prefactor  # this factor is needed in interpolation only
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

## ========================================================================================== ##
## Shared-memory implementation

@kernel function interpolate_to_points_shmem_kernel!(
        vp::NTuple{C, AbstractVector{Z}},
        @Const(gs::NTuple{D}),
        @Const(evalmode::EvaluationMode),
        @Const(points::NTuple{D}),
        @Const(us::NTuple{C, AbstractArray{Z}}),
        @Const(pointperm),
        @Const(cumulative_npoints_per_block::AbstractVector),
        @Const(prefactor::Real),    # = volume of a grid cell = prod(Δxs)
        ::Val{block_dims},
        ::Val{shmem_size},  # this is a bit redundant, but seems to be required for CPU backends (used in tests)
    ) where {C, D, Z <: Number, block_dims, shmem_size}

    @uniform begin
        groupsize = @groupsize()
        nthreads = prod(groupsize)
    end

    block_n = @index(Group, Linear)      # linear index of block
    block_index = @index(Group, NTuple)  # workgroup index (= block index)
    threadidx = @index(Local, Linear)    # in 1:nthreads

    u_local = @localmem(Z, shmem_size)  # allocate shared memory

    # This needs to be in shared memory for CPU tests (otherwise it doesn't survive across
    # @synchronize barriers).
    ishifts_sm = @localmem(Int, D)  # shift between local and global array in each direction

    if threadidx == 1
        # This block (workgroup) will take care of non-uniform points (a + 1):b
        @inbounds for d ∈ 1:D
            ishifts_sm[d] = (block_index[d] - 1) * block_dims[d] + 1
        end
    end

    @synchronize  # synchronise ishifts_sm

    # Interpolate components one by one (to avoid using too much memory)
    @inbounds for c ∈ 1:C
        # Copy grid data from global to shared memory
        M = Kernels.half_support(gs[1])
        gridvalues_to_local_memory!(
            u_local, us[c], ishifts_sm, Val(M);
            threadidx, nthreads,
        )

        @synchronize  # make sure all threads have the same shared data

        # This block will take care of non-uniform points (a + 1):b
        @inbounds a = cumulative_npoints_per_block[block_n]
        @inbounds b = cumulative_npoints_per_block[block_n + 1]

        for i in (a + threadidx):nthreads:b
            # Interpolate at point j
            j = if pointperm === nothing
                i
            else
                @inbounds pointperm[i]
            end

            # TODO: can we do this just once for all components? (if C > 1)
            indvals = ntuple(Val(D)) do d
                @inline
                x = @inbounds points[d][j]
                gdata = Kernels.evaluate_kernel(evalmode, gs[d], x)
                local i₀ = gdata.i - ishifts_sm[d]
                local vals = gdata.values    # kernel values
                # @assert i₀ ≥ 0
                # @assert i₀ + 2M ≤ block_dims_padded[n]
                i₀ => vals
            end

            inds_start = map(first, indvals)
            window_vals = map(last, indvals)
            v = interpolate_from_arrays_shmem(u_local, inds_start, window_vals, prefactor)
            @inbounds vp[c][j] = v
        end
    end

    nothing
end

# Copy values from global to shared memory.
@inline function gridvalues_to_local_memory!(
        u_local::AbstractArray{T, D},
        u_global::AbstractArray{T, D},
        ishifts,
        ::Val{M};
        threadidx, nthreads,
    ) where {T, D, M}
    Ns = size(u_global)
    inds = CartesianIndices(axes(u_local))
    offsets = ntuple(Val(D)) do d
        @inline
        off = ishifts[d] - M
        ifelse(off < 0, off + Ns[d], off)  # make sure the offset is non-negative (to avoid some wrapping below)
    end
    @inbounds for n ∈ threadidx:nthreads:length(inds)
        is = Tuple(inds[n])
        js = ntuple(Val(D)) do d
            @inline
            j = is[d] + offsets[d]
            ifelse(j > Ns[d], j - Ns[d], j)
        end
        u_local[n] = u_global[js...]
    end
    nothing
end

# Interpolate a single "component" (one transform at a time).
# Here vp is a vector instead of a tuple of vectors.
@inline function interpolate_from_arrays_shmem(
        u_local::AbstractArray{Z, D},
        inds_start::NTuple{D},
        window_vals::NTuple{D},
        prefactor,
    ) where {Z, D}
    if @generated
        gprod_init = Symbol(:gprod_, D + 1)  # the name of this variable is important!
        quote
            $gprod_init = prefactor
            v = zero(Z)
            inds = map(eachindex, window_vals)  # = (1:2M, 1:2M, ...)
            @nloops(
                $D, i,
                d -> inds[d],
                d -> begin
                    @inbounds gprod_d = gprod_{d + 1} * window_vals[d][i_d]
                    @inbounds j_d = inds_start[d] + i_d
                end,
                begin
                    js = @ntuple($D, j)
                    @inbounds v += gprod_1 * u_local[js...]
                end,
            )
            v
        end
    else
        v = zero(Z)
        inds = map(eachindex, window_vals)  # = (1:2M, 1:2M, ...)
        @inbounds for I ∈ CartesianIndices(inds)
            gprod = prefactor * prod(ntuple(d -> @inbounds(window_vals[d][I[d]]), Val(D)))
            js = inds_start .+ Tuple(I)
            v += gprod * u_local[js...]
        end
        v
    end
end
