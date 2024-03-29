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
        us::NTuple{C, AbstractArray{T, D}} where {T},
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

function spread_from_points!(
        gs,
        us_all::NTuple{C, AbstractArray},
        x⃗s::AbstractVector,
        vp_all::NTuple{C, AbstractVector},
    ) where {C}
    # Note: the dimensions of arrays have already been checked via check_nufft_nonuniform_data.
    Base.require_one_based_indexing(x⃗s)  # this is to make sure that all indices match
    foreach(Base.require_one_based_indexing, vp_all)
    for i ∈ eachindex(x⃗s)  # iterate over all points
        x⃗ = @inbounds x⃗s[i]
        vs = map(vp -> @inbounds(vp[i]), vp_all)  # non-uniform values at point x⃗
        spread_from_point!(gs, us_all, x⃗, vs)
    end
    us_all
end

function spread_from_point_blocked!(
        gs::NTuple{D, AbstractKernelData},
        us::NTuple{C, AbstractArray{T, D}} where {T},
        x⃗₀::NTuple{D, Number},
        vs::NTuple{C, Number},
        I₀::NTuple,
    ) where {C, D}
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
    # Base.checkbounds.(us, Tuple(Is))  # check that indices fall inside the output array

    vals = map(g -> g.values, gs_eval)
    spread_onto_arrays_blocked!(us, Is, vals, vs)

    us
end

function spread_from_points_blocked!(
        gs,
        bd::BlockData,
        us_all::NTuple{C, AbstractArray},
        xp::AbstractVector,
        vp_all::NTuple{C, AbstractVector},
    ) where {C}
    (; block_dims, pointperm, buffers, indices,) = bd
    Ms = map(Kernels.half_support, gs)
    for us ∈ us_all
        fill!(us, zero(eltype(us)))
    end
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
            for ws ∈ block
                fill!(ws, zero(eltype(ws)))
            end
            for k ∈ (a + 1):b
                l = pointperm[k]
                # @assert bd.blockidx[l] == j  # check that point is really in the current block
                if bd.sort_points === True()
                    x⃗ = xp[k]  # if points have been permuted (may be slightly faster here, but requires permutation in set_points!)
                else
                    x⃗ = xp[l]  # if points have not been permuted
                end
                vs = map(vp -> @inbounds(vp[l]), vp_all)  # values at the non-uniform point x⃗
                spread_from_point_blocked!(gs, block, x⃗, vs, Tuple(I₀))
            end

            # Indices of current block including padding
            Ia = I₀ + oneunit(I₀) - CartesianIndex(Ms)
            Ib = I₀ + CartesianIndex(block_dims) + CartesianIndex(Ms)
            wrap_periodic!(inds_wrapped, Ia, Ib, size(first(us_all)))

            # Add data from block to output array.
            # Note that only one thread can write at a time.
            lock(lck) do
                add_from_block!(us_all, block, inds_wrapped)
            end
        end
    end

    us_all
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

function add_from_block!(
        us_all::NTuple{C, AbstractArray},
        block::NTuple{C, AbstractArray},
        inds_wrapped::Tuple,
    ) where {C}
    for ws ∈ block
        @assert size(ws) == map(length, inds_wrapped)
        Base.require_one_based_indexing(ws)
    end
    for us ∈ us_all
        Base.require_one_based_indexing(us)
    end
    # We explicitly split the first index (fastest) from the other ones.
    # This seems to noticeably improve performance, maybe because the compiler can use SIMD
    # on the innermost loop?
    # Note that performance of this function is critical to get good parallel scaling!
    inds = axes(first(block))
    inds_first, inds_tail = first(inds), Base.tail(inds)
    inds_wrapped_first, inds_wrapped_tail = first(inds_wrapped), Base.tail(inds_wrapped)
    @inbounds for I_tail ∈ CartesianIndices(inds_tail)
        is_tail = Tuple(I_tail)
        js_tail = map(getindex, inds_wrapped_tail, is_tail)
        for i ∈ inds_first
            j = inds_wrapped_first[i]
            for (us, ws) ∈ zip(us_all, block)
                us[j, js_tail...] += ws[i, is_tail...]
            end
        end
    end
    us_all
end

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
        is_tail = map(getindex, imap_tail, js_tail)
        gs_tail = map(getindex, vals_tail, js_tail)
        gprod_tail = prod(gs_tail)
        for j ∈ inds_first
            i = imap_first[j]
            gprod = gprod_tail * vals_first[j]
            for (u, v) ∈ zip(us, vs)
                u[i, is_tail...] += v * gprod
            end
        end
    end
    us
end

# This is basically the same as the non-blocked version, but uses CartesianIndices instead
# of tuples (since indices don't "jump" due to periodic wrapping).
# Moreover, splitting the first index from the other ones seems to improve performance.
function spread_onto_arrays_blocked!(
        us::NTuple{C, AbstractArray{T, D}} where {T},
        Is::CartesianIndices{D},
        vals::NTuple{D, Tuple},
        vs::NTuple{C},
    ) where {C, D}
    inds = map(eachindex, vals)
    inds_first, inds_tail = first(inds), Base.tail(inds)
    vals_first, vals_tail = first(vals), Base.tail(vals)
    @inbounds for J_tail ∈ CartesianIndices(inds_tail)
        js_tail = Tuple(J_tail)
        gs_tail = map(getindex, vals_tail, js_tail)
        gprod_tail = prod(gs_tail)
        for j ∈ inds_first
            I = Is[j, js_tail...]
            gprod = gprod_tail * vals_first[j]
            for (u, v) ∈ zip(us, vs)
                u[I] += v * gprod
            end
        end
    end
    us
end
