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
        spread_from_point!(gs, us, x⃗, v)
    end
    us
end

function spread_from_point_blocked!(gs::NTuple, u::AbstractArray, x⃗₀, v::Number, I₀::NTuple)
    # Evaluate 1D kernels.
    gs_eval = map(Kernels.evaluate_kernel, gs, x⃗₀)

    Ms = map(Kernels.half_support, gs)
    δs = Ms .- I₀  # index offset

    # Determine indices to write in `u` arrays.
    inds = map(gs_eval, gs, δs) do gdata, g, δ
        is = Kernels.kernel_indices(gdata.i, g)  # note: this variant doesn't perform periodic wrapping
        is .+ δ  # shift to beginning of current block
    end
    Is = CartesianIndices(inds)
    # Base.checkbounds(u, Is)  # check that indices fall inside the output array

    vals = map(g -> g.values, gs_eval)
    spread_onto_arrays_blocked!((u,), Is, vals, (v,))

    u
end

function spread_from_points_blocked!(
        gs, bd::BlockData, us::AbstractArray, xp::AbstractVector, vp::AbstractVector,
    )
    (; block_dims, pointperm, buffers, indices,) = bd
    Ms = map(Kernels.half_support, gs)
    fill!(us, zero(eltype(us)))
    Nt = length(buffers)  # usually equal to the number of threads
    # nblocks = length(indices)
    Base.require_one_based_indexing(buffers)
    Base.require_one_based_indexing(indices)
    lck = ReentrantLock()
    Threads.@threads :static for i ∈ 1:Nt
        # j_start = (i - 1) * nblocks ÷ Nt + 1
        # j_end = i * nblocks ÷ Nt
        j_start = bd.blocks_per_thread[i] + 1
        j_end = bd.blocks_per_thread[i + 1]
        block = buffers[i]
        inds_wrapped = bd.buffers_for_indices[i]
        @inbounds for j ∈ j_start:j_end
            a = bd.cumulative_npoints_per_block[j]
            b = bd.cumulative_npoints_per_block[j + 1]
            a == b && continue  # no points in this block (otherwise b > a)

            # Iterate over all points in the current block
            I₀ = indices[j]
            fill!(block, zero(eltype(block)))
            for k ∈ (a + 1):b
                l = pointperm[k]
                # @assert bd.blockidx[l] == j  # check that point is really in the current block
                x⃗ = xp[l]  # if points have not been permuted
                # x⃗ = xp[k]  # if points have been permuted (may be slightly faster here, but requires permutation in sort_points!)
                v = vp[l]
                spread_from_point_blocked!(gs, block, x⃗, v, Tuple(I₀))
            end

            # Indices of current block including padding
            Ia = I₀ + oneunit(I₀) - CartesianIndex(Ms)
            Ib = I₀ + CartesianIndex(block_dims) + CartesianIndex(Ms)
            wrap_periodic!(inds_wrapped, Ia, Ib, size(us))

            # Add data from block to output array.
            # Note that only one thread can write at a time.
            lock(lck) do
                add_from_block!(us, block, inds_wrapped)
            end
        end
    end
    us
end

function wrap_periodic!(inds::NTuple{D}, Ia::CartesianIndex{D}, Ib::CartesianIndex{D}, Ns::Dims{D}) where {D}
    for j ∈ 1:D
        wrap_periodic!(inds[j], Ia[j]:Ib[j], Ns[j])
    end
    inds
end

function wrap_periodic!(inds::AbstractVector, irange::AbstractRange, N)
    for j ∈ eachindex(inds, irange)
        inds[j] = wrap_periodic(irange[j], N)
    end
    inds
end

function wrap_periodic(i::Integer, N::Integer)
    while i ≤ 0
        i += N
    end
    while i > N
        i -= N
    end
    i
end

function add_from_block!(us::AbstractArray, block::AbstractArray, inds_wrapped::Tuple)
    @assert size(block) == map(length, inds_wrapped)
    Base.require_one_based_indexing(us)
    Base.require_one_based_indexing(block)
    @inbounds for I ∈ CartesianIndices(block)
        js = map(getindex, inds_wrapped, Tuple(I))
        us[js...] += block[I]
    end
    us
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

# This is basically the same as the non-blocked version, but uses CartesianIndices instead
# of tuples (since indices don't "jump" due to periodic wrapping).
function spread_onto_arrays_blocked!(
        us::NTuple{C, AbstractArray{T, D}} where {T},
        Is::CartesianIndices{D},
        vals::NTuple{D, Tuple},
        vs::NTuple{C},
    ) where {C, D}
    inds_iter = CartesianIndices(map(eachindex, vals))
    @inbounds for ns ∈ inds_iter  # ns = (ni, nj, ...)
        I = Is[ns]
        gs = map(getindex, vals, Tuple(ns))
        gprod = prod(gs)
        for (u, v) ∈ zip(us, vs)
            u[I] += v * gprod
        end
    end
    us
end
