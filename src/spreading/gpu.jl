# Spread from a single point
@kernel function spread_from_point_naive_kernel!(
        us::NTuple{C},
        @Const(gs::NTuple{D}),
        @Const(evalmode::EvaluationMode),
        @Const(points::NTuple{D}),
        @Const(vp::NTuple{C}),
        @Const(pointperm),
        transform_fold::F,
        callback::Callback,
    ) where {F <: Function, Callback <: Function, C, D}
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
    Ns = spread_actual_dims(Z, Ns_real)  # drop the first dimension (of size 2) if Z <: Complex

    indvals = get_inds_vals_gpu(transform_fold, gs, evalmode, points, Ns, j)

    v⃗ = map(v -> @inbounds(v[j]), vp)
    v⃗_new = @inline callback(v⃗, j)
    spread_onto_arrays_gpu!(us, indvals, v⃗_new, Ns)

    nothing
end

@inline spread_actual_dims(::Type{<:Real}, Ns) = Ns
@inline spread_actual_dims(::Type{<:Complex}, Ns) = Base.tail(Ns)  # actual number of complex elements: drop first dimension

@inline function spread_onto_arrays_gpu!(
        us::NTuple{C, AbstractArray{T}},
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
        # Actually seems to have the same performance as the @generated version (but uses a
        # few more registers).
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
            js_tail = ntuple(Val(D - 1)) do d
                # Determine output index in the current dimension.
                @inline
                j = istart_tail[d] + is_tail[d]
                ifelse(j > Ns_tail[d], j - Ns_tail[d], j)  # periodic wrapping
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
    # The :monotonic is really needed to get decent performance on AMDGPU.
    # On CUDA it doesn't seem to make a difference.
    @inbounds Atomix.@atomic :monotonic u[inds...] += v
    nothing
end

# Atomix.@atomic currently fails for complex data (https://github.com/JuliaGPU/KernelAbstractions.jl/issues/497),
# so the output must be a real array `u`.
@inline function _atomic_add!(u::AbstractArray{T}, v::Complex{T}, inds::Tuple) where {T <: Real}
    @inbounds begin
        Atomix.@atomic :monotonic u[1, inds...] += real(v)
        Atomix.@atomic :monotonic u[2, inds...] += imag(v)
    end
    nothing
end

to_real_array(u::AbstractArray{T}) where {T <: Real} = u
to_real_array(u::AbstractArray{T}) where {T <: Complex} = reinterpret(reshape, real(T), u)  # adds an extra first dimension with size 2

# GPU implementation.
# We assume all arrays are already on the GPU.
function spread_from_points!(
        backend::GPU,
        callback::Callback,
        transform_fold::F,
        bd::Union{BlockDataGPU, NullBlockData},
        gs,
        evalmode::EvaluationMode,
        us::NTuple{C, AbstractGPUArray},
        xp::NTuple{D, AbstractGPUVector},
        vp_all::NTuple{C, AbstractGPUVector};
        cpu_use_atomics = false,  # ignored
    ) where {F <: Function, Callback <: Function, C, D}
    # Note: the dimensions of arrays have already been checked via check_nufft_nonuniform_data.
    foreach(Base.require_one_based_indexing, xp)  # this is to make sure that all indices match
    foreach(Base.require_one_based_indexing, vp_all)

    # Reinterpret `us` as real arrays, in case they are complex.
    # This is to avoid current issues with atomic operations on complex data
    # (https://github.com/JuliaGPU/KernelAbstractions.jl/issues/497).
    us_real = map(to_real_array, us)  # doesn't change anything if data is already real (Z === T)

    pointperm = get_pointperm(bd)                  # nothing in case of NullBlockData
    sort_points = get_sort_points(bd)::StaticBool  # False in the case of NullBlockData

    if pointperm !== nothing
        @assert eachindex(pointperm) == eachindex(xp[1])
    end

    # We use dynamically sized kernels to avoid recompilation, since number of points may
    # change from one call to another.
    ndrange_points = size(xp[1])  # iterate through points
    @assert all(x -> size(x) == ndrange_points, xp)
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
            kernel!(us_real, gs, evalmode, xp, vp_sorted, pointperm_, transform_fold, callback; ndrange)
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
            block_dims_padded = @. block_dims_val + 2M - 1
            shmem_size = block_dims_padded  # dimensions of big shared memory array
            groupsize = groupsize_spreading_gpu_shmem(backend, batch_size_actual)
            ndrange = gpu_shmem_ndrange_from_groupsize(groupsize, ngroups)
            kernel! = spread_from_points_shmem_kernel!(backend, groupsize, ndrange)
            kernel!(
                us_real, gs, evalmode, xp, vp_sorted, pointperm_, bd.cumulative_npoints_per_block,
                HalfSupport(M), block_dims, Val(shmem_size), bd.batch_size, transform_fold, callback,
            )
        end
    end

    if sort_points === True()
        foreach(KA.unsafe_free!, vp_sorted)  # manually deallocate temporary arrays
    end

    us
end

# Determine workgroupsize possibly based on the batch size Np.
# Note that this can be overridden by certain backends (AMDGPU and CUDA can behave quite
# differently).
groupsize_spreading_gpu_shmem(::GPU, Np::Integer) = 256

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
@kernel unsafe_indices=true function spread_from_points_shmem_kernel!(
        us::NTuple{C, AbstractArray{T}},
        @Const(gs::NTuple{D}),
        @Const(evalmode::EvaluationMode),
        @Const(points::NTuple{D}),
        @Const(vp::NTuple{C, AbstractVector{Z}}),
        @Const(pointperm),
        @Const(cumulative_npoints_per_block::AbstractVector),
        ::HalfSupport{M},
        ::Val{block_dims},
        ::Val{shmem_size},  # this is a bit redundant, but seems to be required for CPU backends (used in tests)
        ::Val{Np},  # batch_size
        transform_fold::F,
        callback::Callback,
    ) where {C, D, T, Z, M, Np, block_dims, shmem_size, F <: Function, Callback <: Function}

    @uniform begin
        groupsize = @groupsize()
        nthreads = prod(groupsize)

        # Determine grid dimensions.
        # Note that, to avoid problems with @atomic, we currently work with real-data arrays
        # even when the output is complex. So, if `Z <: Complex`, the first dimension has twice
        # the actual dataset dimensions.
        @assert T === real(Z) # output is a real array (but may actually describe complex data)
        Ns_real = size(first(us))  # dimensions of raw input data
        Ns = spread_actual_dims(Z, Ns_real)  # drop the first dimension (of size 2) if Z <: Complex
    end

    block_n = @index(Group, Linear)      # linear index of block
    block_index = @index(Group, NTuple)  # workgroup index (= block index)
    threadidx = @index(Local, Linear)    # in 1:nthreads

    # Allocate static shared memory
    u_local = @localmem(Z, shmem_size)
    # @assert shmem_size == block_dims .+ (2M - 1)
    @assert T <: Real

    window_vals = @localmem(T, (2M, D, Np))
    inds_start = @localmem(Int, (D, Np))
    vp_sm = @localmem(Z, Np)  # input values copied to shared memory

    # Buffer for indices and lengths.
    # This is needed in CPU version (used in tests), to avoid variables from being
    # "forgotten" after a @synchronize barrier.
    buf_sm = @localmem(Int, 2)
    ishifts_sm = @localmem(Int, D)  # shift between local and global array in each direction

    # This block (workgroup) will take care of non-uniform points (a + 1):b
    @inbounds buf_sm[1] = cumulative_npoints_per_block[block_n]       # = a
    @inbounds buf_sm[2] = cumulative_npoints_per_block[block_n + 1]   # = b
    @inbounds for d ∈ 1:D
        ishifts_sm[d] = (block_index[d] - 1) * block_dims[d] + 1
    end

    # Interpolate components one by one (to avoid using too much memory)
    for c ∈ 1:C
        # Reset shared memory to zero
        for i ∈ threadidx:nthreads:length(u_local)
            @inbounds u_local[i] = 0
        end

        @synchronize  # make sure shared memory is fully set to zero

        # We operate in batches of up to `Np` non-uniform points / sources.
        # The aim is to completely avoid atomic operations on shared memory: instead of
        # parallelising over non-uniform points (1 point = 1 thread), we make all threads
        # work on the same set of points, so that, in the end, all threads have the required
        # information to spread point values onto `u_local`. That spreading operation is
        # done by parallelising over the local grid of dimensions M^D (so that every thread
        # writes to a different part of the array), instead of parallelising over the
        # non-uniform points which would require atomics.

        # The first batch deals with points (a + 1):min(a + Np, b)
        @inbounds for batch_begin in buf_sm[1]:Np:(buf_sm[2] - 1)
            @uniform batch_size = min(Np, buf_sm[2] - batch_begin)  # current batch size

            # (1) Evaluate window functions around each non-uniform point.
            inds = CartesianIndices((1:batch_size, 1:D))  # parallelise over dimensions + points
            @inbounds for n in threadidx:nthreads:length(inds)
                p, d = Tuple(inds[n])
                local i = batch_begin + p  # index of non-uniform point
                local j = if pointperm === nothing
                    i
                else
                    @inbounds pointperm[i]
                end
                g = gs[d]
                x = transform_fold(points[d][j])
                if callback === default_callback
                    # Avoid loading the whole vp[:][j] if we haven't set a callback.
                    vp_sm[p] = vp[c][j]
                else
                    # If we've set a callback, then we need to load all "components" vp[:][j],
                    # since the callback takes a tuple and not a scalar value. This is only
                    # relevant for multiple simultaneous transforms (C = ntransforms > 1).
                    # This is needed, for instance, if we wanted to take a vector product
                    # requiring the whole vp[:][j].
                    let v⃗ = map(v -> @inbounds(v[j]), vp)
                        v⃗_new = @inline callback(v⃗, j)
                        vp_sm[p] = v⃗_new[c]  # only keep component `c` of the output value
                    end
                end
                gdata = Kernels.evaluate_kernel(evalmode, g, x)
                ishift = ishifts_sm[d]
                inds_start[d, p] = gdata.i - ishift
                local vals = gdata.values
                for m ∈ eachindex(vals)
                    window_vals[m, d, p] = vals[m]
                end
            end

            @synchronize  # make sure all threads have the same shared data

            # (2) All threads spread together onto shared memory, avoiding all collisions
            # and thus not requiring atomic operations.
            @inbounds for p in 1:batch_size
                local istart = @view inds_start[:, p]
                local v = vp_sm[p]
                window_vals_p = @view window_vals[:, :, p]
                spread_onto_array_shmem_threads!(u_local, istart, window_vals_p, v; threadidx, nthreads)
                @synchronize  # make sure threads don't write concurrently to the same place (since we don't use atomics)
            end
        end

        # Add values from shared memory onto global memory.
        @inbounds if buf_sm[1] < buf_sm[2]  # skip this step if there were actually no points in this block (if a == b)
            add_from_local_to_global_memory!(
                us[c], u_local, Ns, ishifts_sm, Val(M);
                threadidx, nthreads,
            )
        end

        # Avoid resetting u_local too early in the next iteration (c -> c + 1).
        # This is mostly useful when c < C (but putting an `if` fails...).
        @synchronize
    end

    nothing
end

# Spread a single "component" (one transform at a time).
# This is parallelised across threads in a workgroup.
@inline function spread_onto_array_shmem_threads!(
        u_local::AbstractArray{Z, D},
        inds_start,
        window_vals::AbstractArray{T, 2},  # size (2M, D)
        v::Z;
        threadidx, nthreads,
    ) where {T <: AbstractFloat, D, Z}
    inds = CartesianIndices(ntuple(_ -> axes(window_vals, 1), Val(D)))  # = (1:2M, 1:2M, ...)
    @inbounds for n ∈ threadidx:nthreads:length(inds)
        I = inds[n]
        js = ntuple(Val(D)) do d
            @inline
            @inbounds I[d] + inds_start[d]
        end
        gprod = one(T)
        for d ∈ 1:D
            gprod *= window_vals[I[d], d]
        end
        w = v * gprod
        u_local[js...] += w
    end
    nothing
end

# Add values from shared to global memory.
@inline function add_from_local_to_global_memory!(
        u_global::AbstractArray{T},     # actual data in global array is always real (and may have an extra first dimension)
        u_local::AbstractArray{Z, D},   # shared-memory array can be complex
        Ns::Dims{D},
        ishifts,
        ::Val{M};
        threadidx, nthreads,
    ) where {Z, T, D, M}
    @assert T <: Real  # for atomic operations
    # @assert ndims(u_global) === D + (Z <: Complex)  # extra dimension if data is complex
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
        w = u_local[is...]
        _atomic_add!(u_global, w, js)
    end
    nothing
end

## ==================================================================================================== ##
