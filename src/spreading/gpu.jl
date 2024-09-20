# Spread from a single point
@kernel function spread_from_point_naive_kernel!(
        us::NTuple{C},
        @Const(points::NTuple{D}),
        @Const(vp::NTuple{C}),
        @Const(pointperm),
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
    v⃗ = map(v -> @inbounds(v[j]), vp)

    # Determine grid dimensions.
    Z = eltype(v⃗)
    Ns = size(first(us))
    if Z <: Complex
        @assert eltype(first(us)) <: Real      # output is a real array (but actually describes complex data)
        Ns = Base.setindex(Ns, Ns[1] >> 1, 1)  # actual number of complex elements in first dimension
    end

    # Evaluate 1D kernels.
    gs_eval = map((f, x) -> f(x), evaluate, x⃗)

    # Determine indices to write in `u` arrays.
    inds = map(to_indices, gs_eval, Ns) do f, gdata, N
        f(gdata.i, N)
    end

    vals = map(g -> g.values, gs_eval)

    spread_onto_arrays_gpu!(us, inds, vals, v⃗)

    nothing
end

# Currently the @generated function doesn't seem to speed-up things, but that might change
# if we find a way of avoiding atomic writes (which seem to be the bottleneck here).
@inline function spread_onto_arrays_gpu!(
        us::NTuple{C, AbstractArray{T, D}},
        inds_mapping::NTuple{D, Tuple},
        vals::NTuple{D, NTuple{M, Tg}},
        vs::NTuple{C},
    ) where {T, C, D, M, Tg <: AbstractFloat}
    if @generated
        gprod_init = Symbol(:gprod_, D + 1)  # the name of this variable is important!
        quote
            inds = map(eachindex, inds_mapping)
            $gprod_init = one($Tg)
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
                    @inbounds for n ∈ 1:$C
                        w = vs[n] * gprod_1
                        _atomic_add!(us[n], w, js)
                    end
                end
            )
            nothing
        end
    else
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
                is = (i, is_tail...)
                gprod = gprod_tail * vals_first[j]
                for (u, v) ∈ zip(us, vs)
                    w = v * gprod
                    _atomic_add!(u, w, is)
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

    evaluate = map(Kernels.evaluate_kernel_func, gs)   # kernel evaluation functions
    to_indices = map(Kernels.kernel_indices_func, gs)  # functions returning spreading indices
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
        kernel_perm! = spread_permute_kernel!(backend)
        kernel_perm!(vp_sorted, vp_all, pointperm; workgroupsize, ndrange)
        pointperm_ = nothing  # we don't need any further permutations (all accesses to non-uniform data will be contiguous)
    else
        vp_sorted = vp_all
        pointperm_ = pointperm
    end

    kernel! = spread_from_point_naive_kernel!(backend)
    kernel!(us_real, xs_comp, vp_sorted, pointperm_, evaluate, to_indices; workgroupsize, ndrange)

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
