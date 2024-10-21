using StaticArrays: MVector

# Note: this is also used in spreading
@inline function get_inds_vals_gpu(gs::NTuple{D}, points::NTuple{D}, Ns::NTuple{D}, j::Integer) where {D}
    ntuple(Val(D)) do n
        @inline
        get_inds_vals_gpu(gs[n], points[n], Ns[n], j)
    end
end

# Note: this is also used in spreading
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

    indvals = get_inds_vals_gpu(gs, points, Ns, j)

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
            kernel!(vp_sorted, gs, xs_comp, us, pointperm_, prefactor; ndrange)
        end
    elseif method === :shared_memory
        @assert bd isa BlockDataGPU
        Z = eltype(us[1])
        M = Kernels.half_support(gs[1])
        @assert all(g -> Kernels.half_support(g) === M, gs)  # check that they're all equal
        block_dims_val = block_dims_gpu_shmem(Z, size(us[1]), HalfSupport(M), bd.batch_size)  # this is usually a compile-time constant...
        block_dims = Val(block_dims_val)  # ...which means this doesn't require a dynamic dispatch
        @assert block_dims_val === bd.block_dims
        let ngroups = bd.nblocks_per_dir  # this is the required number of workgroups (number of blocks in CUDA)
            block_dims_padded = @. block_dims_val + 2M - 1  # dimensions of shared memory array
            shmem_size = block_dims_padded
            groupsize = groupsize_shmem(ngroups, block_dims_padded, length(x⃗s))
            ndrange = groupsize .* ngroups
            # TODO:
            # - try out batches in interpolation?
            kernel! = interpolate_to_points_shmem_kernel!(backend, groupsize, ndrange)
            kernel!(
                vp_sorted, gs, xs_comp, us, pointperm_, bd.cumulative_npoints_per_block,
                prefactor, block_dims,
                Val(shmem_size),
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

# Determine groupsize (number of threads per direction) in :shared_memory method.
# Here Np is the number of non-uniform points.
# The distribution of threads across directions mainly (only?) affects the copies between
# shared and global memories, so it can have an important influence of performance due to
# memory accesses.
# NOTE: this is also used in spreading
function groupsize_shmem(ngroups::NTuple{D}, shmem_size::NTuple{D}, Np) where {D}
    # (1) Determine the total number of threads.
    groupsize = 64  # minimum group size should be equal to the warp size (usually 32 on CUDA and 64 on AMDGPU)
    # Increase groupsize if number of points is much larger (at least 4×) than the groupsize.
    # Not sure this is a good idea if the point distribution is very inhomogeneous...
    # TODO see if this improves performance in highly dense cases
    # points_per_group = Np / prod(ngroups)  # average number of points per block
    # while groupsize < 1024 && 4 * groupsize ≤ points_per_group
    #     groupsize *= 2
    # end
    # (2) Determine number of threads in each direction.
    # This mainly affects the performance of global -> shared memory copies.
    # It seems like it's better to parallelise the outer dimensions first.
    gsizes = ntuple(_ -> 1, Val(D))
    p = 1  # product of sizes
    i = D  # parallelise outer dimensions first
    while p < groupsize
        if gsizes[i] < shmem_size[i] || i == 1
            gsizes = Base.setindex(gsizes, gsizes[i] * 2, i)
            p *= 2
        else
            @assert i > 1
            i -= 1
        end
    end
    @assert p == groupsize == prod(gsizes)
    gsizes
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
        @Const(points::NTuple{D}),
        @Const(us::NTuple{C, AbstractArray{Z}}),
        @Const(pointperm),
        @Const(cumulative_npoints_per_block::AbstractVector),
        @Const(prefactor::Real),    # = volume of a grid cell = prod(Δxs)
        ::Val{block_dims},
        ::Val{shmem_size},  # this is a bit redundant, but seems to be required for CPU backends (used in tests)
    ) where {C, D, Z <: Number, block_dims, shmem_size}

    @uniform begin
        groupsize = @groupsize()::Dims{D}
        nthreads = prod(groupsize)
    end

    threadidxs = @index(Local, NTuple)   # in (1:nthreads_x, 1:nthreads_y, ...)
    threadidx = @index(Local, Linear)    # in 1:nthreads
    block_index = @index(Group, NTuple)  # workgroup index (= block index)
    block_n = @index(Group, Linear)      # linear index of block

    u_local = @localmem(Z, shmem_size)  # allocate shared memory

    # Interpolate components one by one (to avoid using too much memory)
    for c ∈ 1:C
        # Copy grid data from global to shared memory
        M = Kernels.half_support(gs[1])
        gridvalues_to_local_memory!(u_local, us[c], Val(M), block_index, block_dims, threadidxs, groupsize)

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
            indvals = ntuple(Val(D)) do n
                @inline
                x = @inbounds points[n][j]
                gdata = Kernels.evaluate_kernel(gs[n], x)
                local vals = gdata.values    # kernel values
                local ishift = (block_index[n] - 1) * block_dims[n] + 1
                local i₀ = gdata.i - ishift
                # @assert i₀ ≥ 0
                # @assert i₀ + 2M ≤ block_dims_padded[n]
                i₀ => vals
            end

            v = interpolate_from_arrays_shmem(u_local, indvals, prefactor)
            @inbounds vp[c][j] = v
        end
    end

    nothing
end

# Copy values from global to shared memory.
# - can we optimise memory access patterns?
@inline function gridvalues_to_local_memory!(
        u_local::AbstractArray{T, D},
        u_global::AbstractArray{T, D},
        ::Val{M}, block_index::Dims{D}, block_dims::Dims{D}, threadidxs::Dims{D}, groupsize::Dims{D},
    ) where {T, D, M}
    if @generated
        quote
            Ns = size(u_global)
            inds = @ntuple($D, n -> axes(u_local, n)[threadidxs[n]:groupsize[n]:end])
            offsets = @ntuple(
                $D,
                n -> let
                    off = (block_index[n] - 1) * block_dims[n] - ($M - 1)
                    ifelse(off < 0, off + Ns[n], off)  # make sure the offset is non-negative (to avoid some wrapping below)
                end
            )
            @nloops(
                $D, i,
                d -> inds[d],
                d -> begin
                    j_d = offsets[d] + i_d
                    j_d = ifelse(j_d > Ns[d], j_d - Ns[d], j_d)
                end,
                begin
                    is = @ntuple($D, i)
                    js = @ntuple($D, j)
                    @inbounds u_local[is...] = u_global[js...]
                end,
            )
            nothing
        end
    else
        Ns = size(u_global)
        inds = ntuple(Val(D)) do n
            axes(u_local, n)[threadidxs[n]:groupsize[n]:end]  # this determines the parallelisation pattern
        end
        offsets = ntuple(Val(D)) do n
            @inline
            off = (block_index[n] - 1) * block_dims[n] - (M - 1)
            ifelse(off < 0, off + Ns[n], off)  # make sure the offset is non-negative (to avoid some wrapping below)
        end
        @inbounds for is ∈ Iterators.product(inds...)
            js = ntuple(Val(D)) do n
                @inline
                j = offsets[n] + is[n]
                ifelse(j > Ns[n], j - Ns[n], j)
            end
            u_local[is...] = u_global[js...]
        end
        nothing
    end
end

# Interpolate a single "component" (one transform at a time).
# Here vp is a vector instead of a tuple of vectors.
@inline function interpolate_from_arrays_shmem(
        u_local::AbstractArray{T, D},
        indvals::NTuple{D},
        prefactor,
    ) where {T, D}
    if @generated
        gprod_init = Symbol(:gprod_, D + 1)  # the name of this variable is important!
        quote
            $gprod_init = prefactor
            v = zero(T)
            inds_start = map(first, indvals)
            vals = map(last, indvals)    # evaluated kernel values in each direction
            inds = map(eachindex, vals)  # = (1:L, 1:L, ...) where L = 2M is the kernel width
            @nloops(
                $D, i,
                d -> inds[d],
                d -> begin
                    @inbounds gprod_d = gprod_{d + 1} * vals[d][i_d]
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
        v = zero(T)
        inds_start = map(first, indvals)
        vals = map(last, indvals)    # evaluated kernel values in each direction
        inds = map(eachindex, vals)  # = (1:L, 1:L, ...) where L = 2M is the kernel width
        @inbounds for I ∈ CartesianIndices(inds)
            gprod = prefactor * prod(ntuple(d -> @inbounds(vals[d][I[d]]), Val(D)))
            js = inds_start .+ Tuple(I)
            v += gprod * u_local[js...]
        end
        v
    end
end
