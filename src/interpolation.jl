function interpolate!(
        gs,
        vp_all::NTuple{C, AbstractVector},
        us::NTuple{C, AbstractArray},
        x⃗s::AbstractVector,
    ) where {C}
    # Note: the dimensions of arrays have already been checked via check_nufft_nonuniform_data.
    Base.require_one_based_indexing(x⃗s)  # this is to make sure that all indices match
    foreach(Base.require_one_based_indexing, vp_all)
    for i ∈ eachindex(x⃗s)  # iterate over all points
        x⃗ = @inbounds x⃗s[i]
        vs = interpolate(gs, us, x⃗) :: NTuple{C}  # non-uniform values at point x⃗
        for (vp, v) ∈ zip(vp_all, vs)
            @inbounds vp[i] = v
        end
    end
    vp_all
end

function interpolate(
        gs::NTuple{D, AbstractKernelData},
        us::NTuple{C, AbstractArray{T, D}} where {T},
        x⃗::NTuple{D},  # coordinates are assumed to be in [0, 2π]
    ) where {D, C}
    @assert C > 0
    map(Base.require_one_based_indexing, us)
    Ns = size(first(us))
    @assert all(u -> size(u) === Ns, us)

    # Evaluate 1D kernels.
    gs_eval = map(Kernels.evaluate_kernel, gs, x⃗)

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

function interpolate_blocked(
        gs::NTuple{D},
        us::NTuple{C, AbstractArray{T, D}} where {T},
        x⃗::NTuple{D},
        I₀::NTuple{D},
    ) where {D, C}
    @assert C > 0
    map(Base.require_one_based_indexing, us)
    Ns = size(first(us))
    @assert all(u -> size(u) === Ns, us)

    # Evaluate 1D kernels.
    gs_eval = map(Kernels.evaluate_kernel, gs, x⃗)

    Ms = map(Kernels.half_support, gs)
    δs = Ms .- I₀  # index offset

    # Determine indices to load from `u` arrays.
    inds = map(gs_eval, gs, δs) do gdata, g, δ
        is = Kernels.kernel_indices(gdata.i, g)  # note: this variant doesn't perform periodic wrapping
        is .+ δ  # shift to beginning of current block
    end
    Is = CartesianIndices(inds)
    # Base.checkbounds(us[1], Is)  # check that indices fall inside the output array

    vals = map(gs_eval, gs) do geval, g
        Δx = gridstep(g)
        geval.values .* Δx
    end

    interpolate_from_arrays_blocked(us, Is, vals)
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
        is_tail = map(getindex, imap_tail, js_tail)
        gs_tail = map(getindex, vals_tail, js_tail)
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

# See add_from_block! and spread_onto_arrays_blocked! for comments on performance.
function interpolate_from_arrays_blocked(
        us::NTuple{C, AbstractArray{T, D}} where {T},
        Is::CartesianIndices{D},
        vals::NTuple{D, Tuple},
    ) where {C, D}
    vs = ntuple(_ -> zero(eltype(first(us))), Val(C))
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
            vs_new = ntuple(Val(C)) do n
                @inline
                @inbounds gprod * us[n][I]
            end
            vs = vs .+ vs_new
        end
    end
    vs
end

function interpolate_blocked!(
        gs,
        bd::BlockData,
        vp_all::NTuple{C, AbstractVector},
        us::NTuple{C, AbstractArray},
        x⃗s::AbstractArray,
    ) where {C}
    (; block_dims, pointperm, buffers, indices,) = bd
    Ms = map(Kernels.half_support, gs)
    Nt = length(buffers)  # usually equal to the number of threads
    # nblocks = length(indices)
    Base.require_one_based_indexing(buffers)
    Base.require_one_based_indexing(indices)

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

            # Indices of current block including padding
            I₀ = indices[j]
            Ia = I₀ + oneunit(I₀) - CartesianIndex(Ms)
            Ib = I₀ + CartesianIndex(block_dims) + CartesianIndex(Ms)
            wrap_periodic!(inds_wrapped, Ia, Ib, size(first(us)))
            copy_to_block!(block, us, inds_wrapped)

            # Iterate over all points in the current block
            for k ∈ (a + 1):b
                l = pointperm[k]
                # @assert bd.blockidx[l] == j  # check that point is really in the current block
                # x⃗ = x⃗s[l]  # if points have not been permuted
                x⃗ = x⃗s[k]  # if points have been permuted (may be slightly faster here, but requires permutation in set_points!)
                vs = interpolate_blocked(gs, block, x⃗, Tuple(I₀)) :: NTuple{C}  # non-uniform values at point x⃗
                for (vp, v) ∈ zip(vp_all, vs)
                    @inbounds vp[l] = v
                end
            end
        end
    end

    vp_all
end

# See add_from_block! for comments on performance.
function copy_to_block!(
        block::NTuple{C, AbstractArray},
        us_all::NTuple{C, AbstractArray},
        inds_wrapped::Tuple,
    ) where {C}
    for ws ∈ block
        @assert size(ws) == map(length, inds_wrapped)
        Base.require_one_based_indexing(ws)
    end
    for us ∈ us_all
        Base.require_one_based_indexing(us)
    end
    inds = axes(first(block))
    inds_first, inds_tail = first(inds), Base.tail(inds)
    inds_wrapped_first, inds_wrapped_tail = first(inds_wrapped), Base.tail(inds_wrapped)
    @inbounds for I_tail ∈ CartesianIndices(inds_tail)
        is_tail = Tuple(I_tail)
        js_tail = map(getindex, inds_wrapped_tail, is_tail)
        for i ∈ inds_first
            j = inds_wrapped_first[i]
            for (us, ws) ∈ zip(us_all, block)
                ws[i, is_tail...] = us[j, js_tail...]
            end
        end
    end
    us_all
end
