# Spread from a single point
@kernel function spread_from_point_naive_kernel!(
        us::NTuple{C},
        @Const(points::NTuple{D}),
        @Const(vp::NTuple{C}),
        evaluate::NTuple{D, <:Function},  # can't be marked Const for some reason
        to_indices::NTuple{D, <:Function},
    ) where {C, D}
    i = @index(Global, Linear)
    x⃗ = map(xs -> @inbounds(xs[i]), points)
    v⃗ = map(v -> @inbounds(v[i]), vp)

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

function spread_onto_arrays_gpu!(
        us::NTuple{C, AbstractArray{T, D}} where {T},
        inds_mapping::NTuple{D, Tuple},
        vals::NTuple{D, Tuple},
        vs::NTuple{C},
    ) where {C, D}
    inds = map(eachindex, inds_mapping)
    @inbounds for J ∈ CartesianIndices(inds)
        js = Tuple(J)
        is = map(inbounds_getindex, inds_mapping, js)
        gs = map(inbounds_getindex, vals, js)
        gprod = prod(gs)
        for (u, v) ∈ zip(us, vs)
            w = v * gprod
            _atomic_add!(u, w, is)
        end
    end
    nothing
end

@inline function _atomic_add!(u::DenseArray{T}, v::T, inds::Tuple) where {T <: Real}
    @inbounds Atomix.@atomic u[inds...] += v
    nothing
end

# Atomix.@atomic currently fails for complex data (https://github.com/JuliaGPU/KernelAbstractions.jl/issues/497),
# so the output must be a real array `u`.
@inline function _atomic_add!(u::DenseArray{T}, v::Complex{T}, inds::Tuple) where {T <: Real}
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

    # TODO: use dynamically sized kernel? (to avoid recompilation, since number of points may change from one call to another)
    ndrange = size(x⃗s)  # iterate through points
    workgroupsize = default_workgroupsize(backend, ndrange)
    kernel! = spread_from_point_naive_kernel!(backend, workgroupsize, ndrange)
    kernel!(us_real, xs_comp, vp_all, evaluate, to_indices)

    us_all
end
