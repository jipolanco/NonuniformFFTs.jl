# Spread from a single point
@kernel function spread_from_point_naive_kernel!(
        us::NTuple{C},
        @Const(gs::NTuple{D}),
        @Const(points::NTuple{D}),
        @Const(vp::NTuple{C}),
        @Const(pointperm),
    ) where {C, D}
    i = @index(Global, Linear)

    j = if pointperm === nothing
        i
    else
        @inbounds pointperm[i]
    end

    # Determine grid dimensions.
    # Note that, to avoid problems with @atomic, we currently work with real-data arrays
    # even when the output is complex. So, if `Z <: Complex`, the first dimension has twice
    # the actual dataset dimensions.
    Z = eltype(vp[1])
    Ns_real = size(first(us))  # dimensions of raw input data
    @assert eltype(first(us)) <: Real # output is a real array (but may actually describe complex data)
    Ns = spread_actual_dims(Z, Ns_real)  # divides the Ns_real[1] by 2 if Z <: Complex

    indvals = get_inds_vals_gpu(gs, points, Ns, j)

    v⃗ = map(v -> @inbounds(v[j]), vp)
    spread_onto_arrays_gpu!(us, indvals, v⃗, Ns)

    nothing
end

@inline spread_actual_dims(::Type{<:Real}, Ns) = Ns
@inline spread_actual_dims(::Type{<:Complex}, Ns) = Base.setindex(Ns, Ns[1] >> 1, 1)  # actual number of complex elements in first dimension

@inline function spread_onto_arrays_gpu!(
        us::NTuple{C, AbstractArray{T, D}},
        indvals::NTuple{D, <:Pair},
        vs::NTuple{C},
        Ns::Dims{D},
    ) where {T, C, D}
    if @generated
        gprod_init = Symbol(:gprod_, D + 1)  # the name of this variable is important!
        Tr = real(T)
        quote
            inds_start = map(first, indvals)  # start of active region in output array
            vals = map(last, indvals)    # evaluated kernel values in each direction
            inds = map(eachindex, vals)  # = (1:L, 1:L, ...) where L = 2M is the kernel width
            $gprod_init = one($Tr)       # product of kernel values (initially 1)
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
                    js = @ntuple($D, j)
                    @inbounds for n ∈ 1:$C
                        w = vs[n] * gprod_1
                        _atomic_add!(us[n], w, js)
                    end
                end
            )
            nothing
        end
    else
        # Fallback implementation in case the @generated version above doesn't work.
        # Actually seems to have the same performance as the @generated version.
        inds_start = map(first, indvals)
        vals = map(last, indvals)
        inds = map(eachindex, vals)
        inds_first, inds_tail = first(inds), Base.tail(inds)
        vals_first, vals_tail = first(vals), Base.tail(vals)
        istart_first, istart_tail = first(inds_start), Base.tail(inds_start)
        N, Ns_tail = first(Ns), Base.tail(Ns)
        @inbounds for I_tail ∈ CartesianIndices(inds_tail)
            is_tail = Tuple(I_tail)
            gs_tail = map(inbounds_getindex, vals_tail, is_tail)
            gprod_tail = prod(gs_tail)
            js_tail = map(istart_tail, is_tail, Ns_tail) do j₀, i, Nloc
                # Determine output index in the current dimension.
                @inline
                j = j₀ + i
                ifelse(j > Nloc, j - Nloc, j)  # periodic wrapping
            end
            for i ∈ inds_first
                j = istart_first + i
                j = ifelse(j > N, j - N, j)  # periodic wrapping
                js = (j, js_tail...)
                gprod = gprod_tail * vals_first[i]
                for (u, v) ∈ zip(us, vs)
                    w = v * gprod
                    _atomic_add!(u, w, js)
                end
            end
        end
    end
    nothing
end

@inline function _atomic_add!(u::AbstractArray{T}, v::T, inds::Tuple) where {T <: Real}
    @inbounds Atomix.@atomic u[inds...] += v
    nothing
end

# Atomix.@atomic currently fails for complex data (https://github.com/JuliaGPU/KernelAbstractions.jl/issues/497),
# so the output must be a real array `u`.
@inline function _atomic_add!(u::AbstractArray{T}, v::Complex{T}, inds::Tuple) where {T <: Real}
    @inbounds begin
        i₁ = 2 * (inds[1] - 1)  # convert from logical index (equivalent complex array) to memory index (real array)
        itail = Base.tail(inds)
        Atomix.@atomic u[i₁ + 1, itail...] += real(v)
        Atomix.@atomic u[i₁ + 2, itail...] += imag(v)
    end
    nothing
end

# GPU implementation.
# We assume all arrays are already on the GPU.
function spread_from_points!(
        backend::GPU,
        bd::Union{BlockDataGPU, NullBlockData},
        gs,
        us::NTuple{C, AbstractGPUArray},
        x⃗s::StructVector,
        vp_all::NTuple{C, AbstractGPUVector},
    ) where {C}
    # Note: the dimensions of arrays have already been checked via check_nufft_nonuniform_data.
    Base.require_one_based_indexing(x⃗s)  # this is to make sure that all indices match
    foreach(Base.require_one_based_indexing, vp_all)

    xs_comp = StructArrays.components(x⃗s)

    # Reinterpret `us` as real arrays, in case they are complex.
    # This is to avoid current issues with atomic operations on complex data
    # (https://github.com/JuliaGPU/KernelAbstractions.jl/issues/497).
    Z = eltype(us[1])
    T = real(Z)
    us_real = if Z <: Real
        us
    else  # complex case
        @assert Z <: Complex
        map(u -> reinterpret(T, u), us)  # note: we don't use reshape, so the first dimension has 2x elements
    end

    pointperm = get_pointperm(bd)                  # nothing in case of NullBlockData
    sort_points = get_sort_points(bd)::StaticBool  # False in the case of NullBlockData

    if pointperm !== nothing
        @assert eachindex(pointperm) == eachindex(x⃗s)
    end

    # We use dynamically sized kernels to avoid recompilation, since number of points may
    # change from one call to another.
    ndrange_points = size(x⃗s)  # iterate through points
    groupsize_points = default_workgroupsize(backend, ndrange_points)

    if sort_points === True()
        vp_sorted = map(similar, vp_all)  # allocate temporary arrays for sorted non-uniform data
        let ndrange = ndrange_points, groupsize = groupsize_points
            kernel_perm! = spread_permute_kernel!(backend, groupsize)
            kernel_perm!(vp_sorted, vp_all, pointperm; ndrange)
        end
        pointperm_ = nothing  # we don't need any further permutations (all accesses to non-uniform data will be contiguous)
    else
        vp_sorted = vp_all
        pointperm_ = pointperm
    end

    method = gpu_method(bd)

    if method === :global_memory
        let ndrange = ndrange_points, groupsize = groupsize_points
            kernel! = spread_from_point_naive_kernel!(backend, groupsize)
            kernel!(us_real, gs, xs_comp, vp_sorted, pointperm_; ndrange)
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
            block_dims_padded = @. block_dims_val + 2M - 1
            shmem_size = block_dims_padded  # dimensions of big shared memory array
            groupsize = groupsize_shmem(ngroups, block_dims_padded, length(x⃗s))
            ndrange = groupsize .* ngroups
            kernel! = spread_from_points_shmem_kernel!(backend, groupsize, ndrange)
            kernel!(
                us_real, gs, xs_comp, vp_sorted, pointperm_, bd.cumulative_npoints_per_block,
                HalfSupport(M), block_dims, Val(shmem_size), bd.batch_size,
            )
        end
    end

    if sort_points === True()
        foreach(KA.unsafe_free!, vp_sorted)  # manually deallocate temporary arrays
    end

    us
end

@kernel function spread_permute_kernel!(vp::NTuple{N}, @Const(vp_in::NTuple{N}), @Const(perm::AbstractVector)) where {N}
    i = @index(Global, Linear)
    j = @inbounds perm[i]
    for n ∈ 1:N
        @inbounds vp[n][i] = vp_in[n][j]
    end
    nothing
end

## ========================================================================================== ##
## Shared-memory implementation

# Process non-uniform points in batches of Np points (or less).
# The idea is to completely avoid slow atomic writes to shared memory arrays (u_local),
# while parallelising some operations across up to Np points.
# TODO: Should Np be equal to the number of threads? or how to choose it optimally so that the block size stays not too small?
@kernel function spread_from_points_shmem_kernel!(
        us::NTuple{C, AbstractArray{T}},
        @Const(gs::NTuple{D}),
        @Const(points::NTuple{D}),
        @Const(vp::NTuple{C, AbstractVector{Z}}),
        @Const(pointperm),
        @Const(cumulative_npoints_per_block::AbstractVector),
        ::HalfSupport{M},
        ::Val{block_dims},
        ::Val{shmem_size},  # this is a bit redundant, but seems to be required for CPU backends (used in tests)
        ::Val{Np},  # batch_size
    ) where {C, D, T, Z, M, Np, block_dims, shmem_size}

    @uniform begin
        groupsize = @groupsize()::Dims{D}
        nthreads = prod(groupsize)

        # Determine grid dimensions.
        # Note that, to avoid problems with @atomic, we currently work with real-data arrays
        # even when the output is complex. So, if `Z <: Complex`, the first dimension has twice
        # the actual dataset dimensions.
        @assert T === real(Z) # output is a real array (but may actually describe complex data)
        Ns_real = size(first(us))  # dimensions of raw input data
        Ns = spread_actual_dims(Z, Ns_real)  # divides the Ns_real[1] by 2 if Z <: Complex
    end

    block_n = @index(Group, Linear)      # linear index of block
    block_index = @index(Group, NTuple)  # workgroup index (= block index)
    threadidx = @index(Local, Linear)    # in 1:nthreads

    # Allocate static shared memory
    u_local = @localmem(Z, shmem_size)
    # @assert shmem_size == block_dims .+ (2M - 1)
    @assert T <: Real
    window_vals = @localmem(T, (2M, D, Np))
    points_sm = @localmem(T, (D, Np))  # points copied to shared memory
    inds_start = @localmem(Int, (D, Np))
    vp_sm = @localmem(Z, Np)  # input values copied to shared memory

    # Buffer for indices and lengths.
    # This is needed in CPU version (used in tests), to avoid variables from being
    # "forgotten" after a @synchronize barrier.
    buf_sm = @localmem(Int, 3)
    ishifts_sm = @localmem(Int, D)  # shift between local and global array in each direction

    if threadidx == 1
        # This block (workgroup) will take care of non-uniform points (a + 1):b
        @inbounds buf_sm[1] = cumulative_npoints_per_block[block_n]       # = a
        @inbounds buf_sm[2] = cumulative_npoints_per_block[block_n + 1]   # = b
        @inbounds for d ∈ 1:D
            ishifts_sm[d] = (block_index[d] - 1) * block_dims[d] + 1
        end
    end

    # Interpolate components one by one (to avoid using too much memory)
    for c ∈ 1:C
        # Reset shared memory to zero
        for i ∈ threadidx:nthreads:length(u_local)
            @inbounds u_local[i] = 0
        end

        @synchronize  # make sure shared memory is fully set to zero

        # The first batch deals with points (a + 1):min(a + Np, b)
        @inbounds for batch_begin in buf_sm[1]:Np:(buf_sm[2] - 1)
            batch_size = min(Np, buf_sm[2] - batch_begin)  # current batch size
            buf_sm[3] = batch_size
            # Iterate over points in the batch (ideally 1 thread per point).
            # Each thread writes to shared memory.
            @inbounds for p in threadidx:nthreads:batch_size
                # Spread from point j
                i = batch_begin + p  # index of non-uniform point
                j = if pointperm === nothing
                    i
                else
                    @inbounds pointperm[i]
                end
                for d ∈ 1:D
                    points_sm[d, p] = points[d][j]
                end
                vp_sm[p] = vp[c][j]
            end

            @synchronize

            # Now evaluate windows associated to each point.
            local inds = CartesianIndices((1:buf_sm[3], 1:D))  # parallelise over dimensions + points
            @inbounds for n in threadidx:nthreads:length(inds)
                p, d = Tuple(inds[n])
                g = gs[d]
                x = points_sm[d, p]
                gdata = Kernels.evaluate_kernel(g, x)
                ishift = ishifts_sm[d]
                inds_start[d, p] = gdata.i - ishift
                local vals = gdata.values
                for m ∈ eachindex(vals)
                    window_vals[m, d, p] = vals[m]
                end
            end

            @synchronize  # make sure all threads have the same shared data

            # Now all threads spread together onto shared memory
            @inbounds for p in 1:buf_sm[3]
                local istart = ntuple(d -> @inbounds(inds_start[d, p]), Val(D))
                local v = vp_sm[p]
                spread_onto_array_shmem_threads!(u_local, istart, window_vals, v, p; threadidx, nthreads)
                @synchronize  # make sure threads don't write concurrently to the same place (since we don't use atomics)
            end
        end

        @synchronize  # make sure we have finished writing to shared memory

        @inbounds if buf_sm[1] < buf_sm[2]  # skip this step if there were actually no points in this block (if a == b)
            add_from_local_to_global_memory!(
                us[c], u_local, Ns, ishifts_sm, Val(M);
                threadidx, nthreads,
            )
        end

        if c < C
            @synchronize  # wait before jumping to next component
        end
    end

    nothing
end

# Spread a single "component" (one transform at a time).
# This is parallelised across threads in a workgroup.
@inline function spread_onto_array_shmem_threads!(
        u_local::AbstractArray{Z, D},
        inds_start::NTuple{D, Integer},
        window_vals::AbstractArray{T, 3},  # static-size shared-memory array (2M, D, Np)
        v::Z, p::Integer;
        threadidx, nthreads,
    ) where {T, D, Z}
    inds = CartesianIndices(ntuple(_ -> axes(window_vals, 1), Val(D)))  # = (1:2M, 1:2M, ...)
    Tr = real(T)
    @inbounds for n ∈ threadidx:nthreads:length(inds)
        I = inds[n]
        js = Tuple(I) .+ inds_start
        gprod = one(Tr)
        for d ∈ 1:D
            gprod *= window_vals[I[d], d, p]
        end
        w = v * gprod
        u_local[js...] += w
    end
    nothing
end

# Add values from shared to global memory.
@inline function add_from_local_to_global_memory!(
        u_global::AbstractArray{T, D},  # actual data in global array is always real
        u_local::AbstractArray{Z, D},   # shared-memory array can be complex
        Ns::Dims{D},
        ishifts,
        ::Val{M};
        threadidx, nthreads,
    ) where {Z, T, D, M}
    @assert T <: Real  # for atomic operations
    inds = CartesianIndices(axes(u_local))
    offsets = ntuple(Val(D)) do d
        @inline
        local off = ishifts[d] - M
        ifelse(off < 0, off + Ns[d], off)  # make sure the offset is non-negative (to avoid some wrapping below)
    end
    @inbounds for n ∈ threadidx:nthreads:length(inds)
        I = inds[n]
        is = Tuple(I)
        js = ntuple(Val(D)) do d
            @inline
            j = is[d] + offsets[d]
            ifelse(j > Ns[d], j - Ns[d], j)
        end
        @inbounds if Z <: Real
            w = u_local[is...]
            Atomix.@atomic u_global[js...] += w
        elseif Z <: Complex
            js_tail = Base.tail(js)
            jfirst = 2 * js[1] - 1
            w = u_local[is...]
            Atomix.@atomic u_global[jfirst, js_tail...] += real(w)
            Atomix.@atomic u_global[jfirst + 1, js_tail...] += imag(w)
        end
    end
    nothing
end

## ==================================================================================================== ##
