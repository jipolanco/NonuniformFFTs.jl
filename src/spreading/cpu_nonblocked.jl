"""
    spread_from_point!(gs::NTuple{D, AbstractKernelData}, evalmode::EvaluationMode, u::AbstractArray{T, D}, x⃗₀, v)

Spread value `v` at point `x⃗₀` onto neighbouring grid points.

The grid is assumed to be periodic with period ``2π`` in each direction.

The point `x⃗₀` **must** be in ``[0, 2π)^D``.
It may be given as a tuple `x⃗₀ = (x₀, y₀, …)` or similarly as a static vector
(from `StaticArrays.jl`).

One can also pass tuples `v = (v₁, v₂, …)` and `u = (u₁, u₂, …)`,
in which case each value `vᵢ` will be spread to its corresponding array `uᵢ`.
This can be useful for spreading vector fields, for instance.
"""
function spread_from_point!(
        gs::NTuple{D, AbstractKernelData},
        evalmode::EvaluationMode,
        us::NTuple{C, AbstractArray{T, D}} where {T},
        x⃗₀::NTuple{D, Number},
        vs::NTuple{C, Number},
    ) where {C, D}
    map(Base.require_one_based_indexing, us)
    Ns = size(first(us))
    @assert all(u -> size(u) === Ns, us)

    # Evaluate 1D kernels.
    gs_eval = map((g, x) -> Kernels.evaluate_kernel(evalmode, g, x), gs, x⃗₀)

    # Determine indices to write in `u` arrays.
    inds = map(gs_eval, gs, Ns) do gdata, g, N
        Kernels.kernel_indices(gdata.i, g, N)
    end

    vals = map(g -> g.values, gs_eval)
    spread_onto_arrays!(us, inds, vals, vs)

    us
end

function spread_from_points!(
        ::CPU,
        callback::Callback,
        transform_fold::F,
        ::NullBlockData,  # no blocking
        gs,
        evalmode::EvaluationMode,
        us_all::NTuple{C, AbstractArray},
        x⃗s::NTuple{N, AbstractVector},
        vp_all::NTuple{C, AbstractVector};
        cpu_use_atomics = false,  # ignored
    ) where {F <: Function, Callback <: Function, C, N}
    # Note: the dimensions of arrays have already been checked via check_nufft_nonuniform_data.
    foreach(Base.require_one_based_indexing, x⃗s)  # this is to make sure that all indices match
    foreach(Base.require_one_based_indexing, vp_all)
    for i ∈ eachindex(x⃗s[1], vp_all[1])  # iterate over all points
        x⃗ = map(xp -> @inbounds(transform_fold(xp[i])), x⃗s)
        vs = map(vp -> @inbounds(vp[i]), vp_all)  # non-uniform values at point x⃗
        vs_new = @inline callback(vs, i)
        spread_from_point!(gs, evalmode, us_all, x⃗, vs_new)
    end
    us_all
end

# TODO: optimise as blocked version, using Base.Cartesian?
function spread_onto_arrays!(
        us::NTuple{C, AbstractArray{T, D}} where {T},
        inds_mapping::NTuple{D, Tuple},
        vals::NTuple{D, Tuple},
        vs::NTuple{C},
    ) where {C, D}
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
            for (u, v) ∈ zip(us, vs)
                w = v * gprod
                u[i, is_tail...] += w
            end
        end
    end
    us
end
