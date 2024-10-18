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
        @Const(cumulative_npoints_per_block::AbstractVector),
        @Const(Δxs::NTuple{D}),           # grid step in each direction (oversampled grid)
        ::Val{block_dims},
    ) where {C, D, block_dims}
    groupsize = @groupsize()::Dims{D}
    nthreads = prod(groupsize)
    threadidxs = @index(Local, NTuple)   # in (1:nthreads_x, 1:nthreads_y, ...)
    threadidx = @index(Local, Linear)    # in 1:nthreads
    block_index = @index(Group, NTuple)  # workgroup index (= block index)
    block_n = @index(Group, Linear)      # linear index of block

    # TODO: support C > 1 (do one component at a time, to use less memory?)
    T = eltype(us[1])
    M = Kernels.half_support(gs[1])  # assume they're all equal
    block_dims_padded = @. block_dims + 2M - 1
    u_local = @localmem(T, block_dims_padded)  # allocate shared memory

    # This block will take care of non-uniform points (a + 1):b
    @inbounds a = cumulative_npoints_per_block[block_n]
    @inbounds b = cumulative_npoints_per_block[block_n + 1]
    ΔV = prod(Δxs)

    # Shift from indices in global array to indices in local array
    idx_shift = @. (block_index - 1) * block_dims + 1

    # Interpolate components one by one (to avoid using too much memory)
    for c ∈ 1:C
        # Copy grid data from global to shared memory
        gridvalues_to_local_memory!(u_local, us[c], Val(M), block_index, block_dims, threadidxs, groupsize)

        @synchronize  # make sure all threads have the same shared data

        for i in (a + threadidx):nthreads:b
            # Interpolate at point j
            j = if pointperm === nothing
                i
            else
                @inbounds pointperm[i]
            end

            indvals = ntuple(Val(D)) do n
                @inline
                x = @inbounds points[n][j]
                gdata = Kernels.evaluate_kernel(gs[n], x)
                local vals = gdata.values    # kernel values
                local M = Kernels.half_support(gs[n])
                local i₀ = gdata.i - idx_shift[n]
                @assert i₀ ≥ 0
                @assert i₀ + 2M ≤ block_dims_padded[n]
                i₀ => vals
            end

            inds_start = map(first, indvals)
            vals = map(last, indvals)    # evaluated kernel values in each direction
            inds = map(eachindex, vals)  # = (1:L, 1:L, ...) where L = 2M is the kernel width

            gprod_4 = ΔV
            v = zero(T)

            @inbounds for i_3 in inds[3]
                gprod_3 = gprod_4 * vals[3][i_3]
                j_3 = inds_start[3] + i_3
                for i_2 in inds[2]
                    gprod_2 = gprod_3 * vals[2][i_2]
                    j_2 = inds_start[2] + i_2
                    for i_1 in inds[1]
                        gprod_1 = gprod_2 * vals[1][i_1]
                        j_1 = inds_start[1] + i_1
                        js = (j_1, j_2, j_3)
                        v += gprod_1 * u_local[js...]
                    end
                end
            end

            @inbounds vp[c][j] = v
        end
    end

    nothing
end

# TODO
# - generalise to all D
# - parallelise over multiple directions?
# - is it possible to optimise the memory access patterns?
@inline function gridvalues_to_local_memory!(
        u_local::AbstractArray{T, D},
        u_global::AbstractArray{T, D},
        ::Val{M},
        block_index::Dims{D}, block_dims::Dims{D},
        threadidxs::Dims{D},
        groupsize::Dims{D},
    ) where {T, D, M}
    Ns = size(u_global)
    @assert D == 3
    inds_1 = axes(u_local, 1)[threadidxs[1]:groupsize[1]:end]
    inds_2 = axes(u_local, 2)[threadidxs[2]:groupsize[2]:end]
    inds_3 = axes(u_local, 3)[threadidxs[3]:groupsize[3]:end]
    offsets = @. (block_index - 1) * block_dims - (M - 1)
    @inbounds for i_3 in inds_3
        j_3 = offsets[3] + i_3
        j_3 = ifelse(j_3 ≤ 0, j_3 + Ns[3], j_3)
        j_3 = ifelse(j_3 > Ns[3], j_3 - Ns[3], j_3)
        for i_2 in inds_2
            j_2 = offsets[2] + i_2
            j_2 = ifelse(j_2 ≤ 0, j_2 + Ns[2], j_2)
            j_2 = ifelse(j_2 > Ns[2], j_2 - Ns[2], j_2)
            for i_1 in inds_1
                j_1 = offsets[1] + i_1
                j_1 = ifelse(j_1 ≤ 0, j_1 + Ns[1], j_1)
                j_1 = ifelse(j_1 > Ns[1], j_1 - Ns[1], j_1)
                is = (i_1, i_2, i_3)
                js = (j_1, j_2, j_3)
                u_local[is...] = u_global[js...]  # TODO: use linear index instead of is?
            end
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
            groupsize = groupsize_shmem(ngroups, length(x⃗s))
            ndrange = groupsize .* ngroups
            kernel! = interpolate_to_points_shmem_kernel!(backend, groupsize, ndrange)
            kernel!(vp_sorted, gs, xs_comp, us, pointperm_, bd.cumulative_npoints_per_block, Δxs, block_dims)
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

# Determine groupsize (number of threads per direction) in :shared_memory method.
# Here Np is the number of non-uniform points.
function groupsize_shmem(ngroups::NTuple{D}, Np) where {D}
    points_per_group = Np / prod(ngroups)  # average number of points per block
    groupsize = 64  # minimum group size should be equal to the warp size (usually 32 on CUDA and 64 on AMDGPU)
    # Increase groupsize if number of points is much larger (at least 4×) than the groupsize.
    # (Not sure this is a good idea if the point distribution is very inhomogeneous...)
    # TODO: maybe leave it at 64?
    while groupsize < 1024 && 4 * groupsize ≤ points_per_group
        groupsize *= 2
    end
    gsizes = ntuple(_ -> 1, Val(D))
    p = 1  # product of sizes
    i = 1
    while p < groupsize
        gsizes = Base.setindex(gsizes, gsizes[i] * 2, i)
        p *= 2
        i = mod1(i + 1, D)
    end
    @assert p == groupsize == prod(gsizes)
    gsizes
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
