"""
    spread_from_point!(gs::NTuple{D, AbstractKernelData}, u::AbstractArray{T, D}, x⃗₀, v)

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
        us::NTuple{C, AbstractArray{T,D}} where {T},
        x⃗₀::NTuple{D, Number},
        vs::NTuple{C, Number},
    ) where {C, D}
    map(Base.require_one_based_indexing, us)
    Ns = size(first(us))
    @assert all(u -> size(u) === Ns, us)

    # Evaluate 1D kernels.
    gs_eval = map(Kernels.evaluate_kernel, gs, x⃗₀)

    # Determine indices to write in `u` arrays.
    inds = map(gs_eval, gs, Ns) do gdata, g, N
        Kernels.kernel_indices(gdata.i, g, N)
    end

    vals = map(g -> g.values, gs_eval)
    spread_onto_arrays!(us, inds, vals, vs)

    us
end

function spread_from_point!(gs::NTuple, u::AbstractArray, x⃗₀, v::Number)
    spread_from_point!(gs, (u,), x⃗₀, (v,))
end

function spread_from_points!(gs, us, x⃗s::AbstractVector, vs::AbstractVector)
    for (x⃗, v) ∈ zip(x⃗s, vs)
        y⃗ = to_unit_cell(x⃗)  # fold coordinates to [0, 2π] unit cell
        spread_from_point!(gs, us, y⃗, v)
    end
    us
end

to_unit_cell(x⃗) = map(_to_unit_cell, x⃗)

function _to_unit_cell(x::Real)
    L = oftype(x, 2π)
    while x < 0
        x += L
    end
    while x ≥ L
        x -= L
    end
    x
end

function spread_onto_arrays!(
        us::NTuple{C, AbstractArray{T, D}} where {T},
        inds::NTuple{D, Tuple},
        vals::NTuple{D, Tuple},
        vs::NTuple{C},
    ) where {C, D}
    inds_iter = CartesianIndices(map(eachindex, inds))
    @inbounds for ns ∈ inds_iter  # ns = (ni, nj, ...)
        is = map(getindex, inds, Tuple(ns))
        gs = map(getindex, vals, Tuple(ns))
        gprod = prod(gs)
        for (u, v) ∈ zip(us, vs)
            u[is...] += v * gprod
        end
    end
    us
end

