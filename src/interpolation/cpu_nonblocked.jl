function interpolate!(
        backend::CPU,
        ::NullBlockData,
        gs,
        evalmode::EvaluationMode,
        vp_all::NTuple{C, AbstractVector},
        us::NTuple{C, AbstractArray},
        x⃗s::AbstractVector,
    ) where {C}
    # Note: the dimensions of arrays have already been checked via check_nufft_nonuniform_data.
    Base.require_one_based_indexing(x⃗s)  # this is to make sure that all indices match
    foreach(Base.require_one_based_indexing, vp_all)
    for i ∈ eachindex(x⃗s)  # iterate over all points
        x⃗ = @inbounds x⃗s[i]
        vs = interpolate(gs, evalmode, us, x⃗) :: NTuple{C}  # non-uniform values at point x⃗
        for (vp, v) ∈ zip(vp_all, vs)
            @inbounds vp[i] = v
        end
    end
    vp_all
end

function interpolate(
        gs::NTuple{D, AbstractKernelData},
        evalmode::EvaluationMode,
        us::NTuple{C, AbstractArray{T, D}} where {T},
        x⃗::NTuple{D},  # coordinates are assumed to be in [0, 2π]
    ) where {D, C}
    @assert C > 0
    map(Base.require_one_based_indexing, us)
    Ns = size(first(us))
    @assert all(u -> size(u) === Ns, us)

    # Evaluate 1D kernels.
    gs_eval = map((g, x) -> Kernels.evaluate_kernel(evalmode, g, x), gs, x⃗)

    # Determine indices to load from `u` arrays.
    inds = map(gs_eval, gs, Ns) do gdata, g, N
        Kernels.kernel_indices(gdata.i, g, N)
    end

    vals = map(gs_eval, gs) do geval, g
        Δx = gridstep(g)
        geval.values .* Δx
    end

    interpolate_from_arrays(us, inds, vals)
end

function interpolate_from_arrays(
        us::NTuple{C, AbstractArray{T, D}} where {T},
        inds_mapping::NTuple{D, Tuple},
        vals::NTuple{D, Tuple},
    ) where {C, D}
    vs = ntuple(_ -> zero(eltype(first(us))), Val(C))
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
            gprod = gprod_tail * vals_first[j]
            vs_new = ntuple(Val(C)) do n
                @inline
                @inbounds gprod * us[n][i, is_tail...]
            end
            vs = vs .+ vs_new
        end
    end
    vs
end
