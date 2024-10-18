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

@inline spread_real_dims(::Type{<:Real}, Ns) = Ns
@inline spread_real_dims(::Type{<:Complex}, Ns) = Base.setindex(Ns, Ns[1] << 1, 1)

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
        block_dims_val = block_dims_gpu_shmem(Z, size(us[1]), HalfSupport(M))  # this is usually a compile-time constant...
        block_dims = Val(block_dims_val)  # ...which means this doesn't require a dynamic dispatch
        @assert block_dims_val === bd.block_dims
        let ngroups = bd.nblocks_per_dir  # this is the required number of workgroups (number of blocks in CUDA)
            block_dims_padded = @. block_dims_val + 2M - 1  # dimensions of shared memory array
            shmem_size = spread_real_dims(Z, block_dims_padded)
            groupsize = groupsize_shmem(ngroups, block_dims_padded, length(x⃗s))
            ndrange = groupsize .* ngroups
            kernel! = spread_from_points_shmem_kernel!(backend, groupsize, ndrange)
            kernel!(
                us_real, gs, xs_comp, vp_sorted, pointperm_,
                bd.cumulative_npoints_per_block, block_dims,
                Val(shmem_size),
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

## ==================================================================================================== ##
## Shared-memory implementation

@kernel function spread_from_points_shmem_kernel!(
        us::NTuple{C, AbstractArray{T}},
        @Const(gs::NTuple{D, AbstractKernelData{<:Any, M}}),
        @Const(points::NTuple{D}),
        @Const(vp::NTuple{C, AbstractVector{Z}}),
        @Const(pointperm),
        @Const(cumulative_npoints_per_block::AbstractVector),
        ::Val{block_dims},
        ::Val{shmem_size},  # this is a bit redundant, but seems to be required for CPU backends (used in tests)
    ) where {C, D, T <: AbstractFloat, Z <: Number, M, block_dims, shmem_size}

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
    threadidxs = @index(Local, NTuple)   # in (1:nthreads_x, 1:nthreads_y, ...)
    threadidx = @index(Local, Linear)    # in 1:nthreads

    u_local = @localmem(T, shmem_size)  # allocate shared memory

    # Interpolate components one by one (to avoid using too much memory)
    for c ∈ 1:C
        # Reset shared memory to zero
        for i ∈ threadidx:nthreads:length(u_local)
            @inbounds u_local[i] = 0
        end

        @synchronize  # make sure shared memory is fully set to zero

        # This block will take care of non-uniform points (a + 1):b
        @inbounds a = cumulative_npoints_per_block[block_n]
        @inbounds b = cumulative_npoints_per_block[block_n + 1]

        for i in (a + threadidx):nthreads:b
            # Spread from point j
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

            @inbounds v = vp[c][j]
            spread_onto_array_shmem!(u_local, indvals, v)
        end

        @synchronize  # make sure we have finished writing to shared memory

        add_from_local_to_global_memory!(
            Z, us[c], u_local, Ns, Val(M), block_index,
            block_dims, threadidxs, groupsize,
        )

        if c < C
            @synchronize  # wait before jumping to next component
        end
    end

    nothing
end

# Spread a single "component" (one transform at a time).
@inline function spread_onto_array_shmem!(
        u_local::AbstractArray{T, D},
        indvals::NTuple{D},
        v::Z,
    ) where {T, D, Z}
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
                    @inbounds gprod_d = gprod_{d + 1} * vals[d][i_d]
                    @inbounds j_d = inds_start[d] + i_d
                end,
                begin
                    js = @ntuple($D, j)
                    w = v * gprod_1
                    _atomic_add!(u_local, w, js)  # this is slow!! (atomics in shared memory) -- is it Atomix's fault?
                end,
            )
            v
        end
    else
        aaa  # TODO: implement
    end
end

# Add values from shared to global memory.
@inline function add_from_local_to_global_memory!(
        ::Type{Z},  # "logical" type of data (e.g. ComplexF64)
        u_global::AbstractArray{T, D},  # actual data is always real
        u_local::AbstractArray{T, D},
        Ns::Dims{D},
        ::Val{M}, block_index::Dims{D}, block_dims::Dims{D}, threadidxs::Dims{D}, groupsize::Dims{D},
    ) where {Z, T, D, M}
    if @generated
        @assert T <: Real  # for atomic operations
        skip = sizeof(Z) ÷ sizeof(T)  # 1 (Z real) or 2 (Z complex)
        quote
            # @assert skip ∈ (1, 2)
            skips = @ntuple($D, n -> n == 1 ? $skip : 1)
            inds = @ntuple($D, n -> axes(u_local, n)[1:(end ÷ skips[n])][threadidxs[n]:groupsize[n]:end])
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
                    @inbounds if $skip === 1  # real data (Z <: Real)
                        w = u_local[is...]
                        Atomix.@atomic u_global[js...] += w
                    elseif $skip === 2  # complex data (Z <: Complex)
                        is_tail = @ntuple($(D - 1), d -> i_{d + 1})
                        js_tail = @ntuple($(D - 1), d -> j_{d + 1})
                        ifirst = 2 * i_1 - 1
                        jfirst = 2 * j_1 - 1
                        w = u_local[ifirst, is_tail...]  # real part
                        Atomix.@atomic u_global[jfirst, js_tail...] += w
                        w = u_local[ifirst + 1, is_tail...]  # imaginary part
                        Atomix.@atomic u_global[jfirst + 1, js_tail...] += w
                    end
                end,
            )
            nothing
        end
    else
        aaa  # TODO: implement
    end
end

## ==================================================================================================== ##
