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

    x⃗ = map(xs -> @inbounds(xs[j]), points)
    v⃗ = map(v -> @inbounds(v[j]), vp)

    # Determine grid dimensions.
    # Note that, to avoid problems with @atomic, we currently work with real-data arrays
    # even when the output is complex. So, if `Z <: Complex`, the first dimension has twice
    # the actual dataset dimensions.
    Z = eltype(v⃗)
    Ns_real = size(first(us))  # dimensions of raw input data
    @assert eltype(first(us)) <: Real # output is a real array (but may actually describe complex data)
    Ns = spread_actual_dims(Z, Ns_real)  # divides the Ns_real[1] by 2 if Z <: Complex

    # Evaluate 1D kernels.
    gs_eval = map(Kernels.evaluate_kernel, gs, x⃗)

    # Determine indices to write in `u` arrays.
    indvals = ntuple(Val(D)) do n
        @inbounds begin
            gdata = gs_eval[n]
            vals = gdata.values
            M = Kernels.half_support(gs[n])
            i₀ = gdata.i - M  # active region is (i₀ + 1):(i₀ + 2M) (up to periodic wrapping)
            i₀ = ifelse(i₀ < 0, i₀ + Ns[n], i₀)  # make sure i₀ ≥ 0
            i₀ => vals
        end
    end

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
            $gprod_init = one($Tr)       # product of kernel values
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
        us_all::NTuple{C, AbstractGPUArray},
        x⃗s::StructVector,
        vp_all::NTuple{C, AbstractGPUVector},
    ) where {C}
    # Note: the dimensions of arrays have already been checked via check_nufft_nonuniform_data.
    Base.require_one_based_indexing(x⃗s)  # this is to make sure that all indices match
    foreach(Base.require_one_based_indexing, vp_all)

    xs_comp = StructArrays.components(x⃗s)

    # Reinterpret `us_all` as real arrays, in case they are complex.
    # This is to avoid current issues with atomic operations on complex data
    # (https://github.com/JuliaGPU/KernelAbstractions.jl/issues/497).
    Z = eltype(us_all[1])
    T = real(Z)
    us_real = if Z <: Real
        us_all
    else  # complex case
        @assert Z <: Complex
        map(u -> reinterpret(T, u), us_all)  # note: we don't use reshape, so the first dimension has 2x elements
    end

    pointperm = get_pointperm(bd)                  # nothing in case of NullBlockData
    sort_points = get_sort_points(bd)::StaticBool  # False in the case of NullBlockData

    if pointperm !== nothing
        @assert eachindex(pointperm) == eachindex(x⃗s)
    end

    # We use dynamically sized kernels to avoid recompilation, since number of points may
    # change from one call to another.
    ndrange = size(x⃗s)  # iterate through points
    workgroupsize = default_workgroupsize(backend, ndrange)

    if sort_points === True()
        vp_sorted = map(similar, vp_all)  # allocate temporary arrays for sorted non-uniform data
        kernel_perm! = spread_permute_kernel!(backend, workgroupsize)
        kernel_perm!(vp_sorted, vp_all, pointperm; ndrange)
        pointperm_ = nothing  # we don't need any further permutations (all accesses to non-uniform data will be contiguous)
    else
        vp_sorted = vp_all
        pointperm_ = pointperm
    end

    kernel! = spread_from_point_naive_kernel!(backend, workgroupsize)
    kernel!(us_real, gs, xs_comp, vp_sorted, pointperm_; ndrange)

    if sort_points === True()
        foreach(KA.unsafe_free!, vp_sorted)  # manually deallocate temporary arrays
    end

    us_all
end

@kernel function spread_permute_kernel!(vp::NTuple{N}, @Const(vp_in::NTuple{N}), @Const(perm::AbstractVector)) where {N}
    i = @index(Global, Linear)
    j = @inbounds perm[i]
    for n ∈ 1:N
        @inbounds vp[n][i] = vp_in[n][j]
    end
    nothing
end
