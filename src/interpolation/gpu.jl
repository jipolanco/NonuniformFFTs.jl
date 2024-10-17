using StaticArrays: MVector

# TODO: use this also in spreading, and move to separate file
@inline function get_inds_vals_gpu(gs::NTuple{D}, points::NTuple{D}, Ns::NTuple{D}, j::Integer) where {D}
    ntuple(Val(D)) do n
        @inline
        get_inds_vals_gpu(gs[n], points[n], Ns[n], j)
    end
end

@inline function get_inds_vals_gpu(g::AbstractKernelData, points::AbstractVector, N::Integer, j::Integer)
    x = @inbounds points[j]
    gdata = Kernels.evaluate_kernel(g, x)
    vals = gdata.values    # kernel values
    M = Kernels.half_support(g)
    i₀ = gdata.i - M  # active region is (i₀ + 1):(i₀ + 2M) (up to periodic wrapping)
    i₀ = ifelse(i₀ < 0, i₀ + N, i₀)  # make sure i₀ ≥ 0
    i₀ => vals
end

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

    # Determine grid dimensions.
    # Unlike in spreading, here `us` can be made of arrays of complex numbers, because we
    # don't perform atomic operations. This is why the code is simpler here.
    Ns = size(first(us))  # grid dimensions

    indvals = get_inds_vals_gpu(gs, points, Ns, j)

    v⃗ = interpolate_from_arrays_gpu(us, indvals, Ns, Δxs)

    for n ∈ eachindex(vp, v⃗)
        @inbounds vp[n][j] = v⃗[n]
    end

    nothing
end

@kernel function interpolate_to_points_shmem_kernel!(
        vp::NTuple{C},
        @Const(gs::NTuple{D}),
        @Const(points::NTuple{D}),
        @Const(us::NTuple{C}),
        @Const(pointperm),
        @Const(Δxs::NTuple{D}),           # grid step in each direction (oversampled grid)
        ::Val{block_dims},
    ) where {C, D, block_dims}
    is_kernel = @index(Global, NTuple)::NTuple{D}
    groupsize = @groupsize()::NTuple{D}  # generally equal to (nthreads, 1, 1, ...)
    threadidx = @index(Local, Linear)
    block_index = is_kernel ./ groupsize
    nthreads = prod(groupsize)

    # TODO: support C > 1 (do one component at a time, to use less memory?)
    T = eltype(us[1])
    M = Kernels.half_support(gs[1])  # assume they're all equal
    block_dims_padded = block_dims .+ 2M
    u_local = @localmem(T, block_dims_padded)  # allocate shared memory

    # Copy grid data from global to shared memory
    gridvalues_to_local_memory!(u_local, us[1], Val(M), block_index, block_dims, threadidx, nthreads)

    @synchronize  # make sure all threads have the same shared data

    nothing
end

@inline function gridvalues_to_local_memory!(
        u_local::AbstractArray{T, D},
        u_global::AbstractArray{T, D},
        ::Val{M},
        block_index::NTuple{D}, block_dims::NTuple{D},
        threadidx::Integer,
        nthreads::Integer,
    ) where {T, D, M}
    Ns = size(u_global)
    # inds = ntuple(Val(D)) do n
    #     @inline
    #     a = (block_index[n] - 1) * block_dims[n] + 1  # start of current block in global grid (excluding ghost cells)
    #     b = block_index[n] * block_dims[n]            # end of current block
    #     (a - M):(b + M)  # range including ghost cells
    # end
    # TODO: split work across threads
    # Assume D = 3 for now
    if threadidx == 1
        # for is in iter
        for i_3 in axes(u_local, 3), i_2 in axes(u_local, 2), i_1 in axes(u_local, 1)
            # For some reason, type assertions are needed for things to work on AMDGPU
            j_1 = mod1(block_index[1] - 1 * block_dims[1] + i_1 - M, Ns[1])::Int
            j_2 = mod1(block_index[2] - 1 * block_dims[2] + i_2 - M, Ns[2])::Int
            j_3 = mod1(block_index[3] - 1 * block_dims[3] + i_3 - M, Ns[3])::Int
            is = (i_1, i_2, i_3)::Dims{D}
            js = (j_1, j_2, j_3)::Dims{D}
            u_local[is...] = u_global[js...]
        end
    end
    nothing
end

function interpolate!(
        backend::GPU,
        bd::Union{BlockDataGPU, NullBlockData},
        gs::NTuple{D},
        vp_all::NTuple{C, AbstractVector},
        us::NTuple{C, AbstractArray},
        x⃗s::AbstractVector,
    ) where {C, D}
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

    method = gpu_method(bd)

    if method === :global_memory
        # We use dynamically sized kernels to avoid recompilation, since number of points may
        # change from one call to another.
        let ndrange = size(x⃗s)  # iterate through points
            groupsize = default_workgroupsize(backend, ndrange)
            kernel! = interpolate_to_point_naive_kernel!(backend, groupsize)
            kernel!(vp_sorted, gs, xs_comp, us, pointperm_, Δxs; ndrange)
        end
    elseif method === :shared_memory
        @assert bd isa BlockDataGPU
        Z = eltype(us[1])
        M = Kernels.half_support(gs[1])
        @assert all(g -> Kernels.half_support(g) === M, gs)  # check that they're all equal
        block_dims_val = block_dims_gpu_shmem(Z, size(us[1]), HalfSupport(M))  # this is usually a compile-time constant...
        block_dims = Val(block_dims_val)  # ...which means this doesn't require a dynamic dispatch
        @assert block_dims_val === bd.block_dims
        let ngroups = bd.nblocks_per_dir  # this is the required number of workgroups (number of blocks in CUDA)
            groupsize = 64  # TODO: this should roughly be the number of non-uniform points per block (or less)
            groupsize_dims = ntuple(d -> d == 1 ? groupsize : 1, D)  # "augment" first dimension
            ndrange = groupsize_dims .* ngroups
            kernel! = interpolate_to_points_shmem_kernel!(backend, groupsize_dims, ndrange)
            kernel!(vp_sorted, gs, xs_comp, us, pointperm_, Δxs, block_dims)
        end
    end

    if sort_points === True()
        let groupsize = default_workgroupsize(backend, ndrange)
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
