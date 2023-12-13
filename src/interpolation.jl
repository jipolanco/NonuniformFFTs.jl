function interpolate!(gs, vp::AbstractArray, us, xp::AbstractArray)
    @assert axes(vp) === axes(xp)
    for i ∈ eachindex(vp)
        vp[i] = interpolate(gs, us, xp[i])
    end
    vp
end

function interpolate(
        gs::NTuple{D, AbstractKernelData},
        us::NTuple{M, AbstractArray{T, D}} where {T},
        x⃗::NTuple{D},  # coordinates are assumed to be in [0, 2π]
    ) where {D, M}
    @assert M > 0
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

interpolate(gs::NTuple, u::AbstractArray, x⃗) = only(interpolate(gs, (u,), x⃗))

function interpolate_blocked(
        gs::NTuple{D},
        us::NTuple{M, AbstractArray{T, D}} where {T},
        x⃗::NTuple{D},
        I₀::NTuple{D},
    ) where {D, M}
    @assert M > 0
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

interpolate_blocked(gs::NTuple, u::AbstractArray, args...) = only(interpolate_blocked(gs, (u,), args...))

function interpolate_from_arrays(
        us::NTuple{C, AbstractArray{T, D}} where {T},
        inds::NTuple{D, Tuple},
        vals::NTuple{D, Tuple},
    ) where {C, D}
    vs = ntuple(_ -> zero(eltype(first(us))), Val(C))
    inds_iter = CartesianIndices(map(eachindex, inds))
    @inbounds for ns ∈ inds_iter  # ns = (ni, nj, ...)
        is = map(getindex, inds, Tuple(ns))
        gs = map(getindex, vals, Tuple(ns))
        gprod = prod(gs)
        vs_new = ntuple(Val(C)) do j
            @inline
            gprod * us[j][is...]
        end
        vs = vs .+ vs_new
    end
    vs
end

function interpolate_from_arrays_blocked(
        us::NTuple{C, AbstractArray{T, D}} where {T},
        Is::CartesianIndices{D},
        vals::NTuple{D, Tuple},
    ) where {C, D}
    vs = ntuple(_ -> zero(eltype(first(us))), Val(C))
    inds_iter = CartesianIndices(map(eachindex, vals))
    @inbounds for ns ∈ inds_iter  # ns = (ni, nj, ...)
        I = Is[ns]
        gs = map(getindex, vals, Tuple(ns))
        gprod = prod(gs)
        vs_new = ntuple(Val(C)) do j
            @inline
            gprod * us[j][I]
        end
        vs = vs .+ vs_new
    end
    vs
end

function interpolate_blocked!(gs, blocks::BlockData, vp::AbstractArray, us, xp::AbstractArray)
    @assert axes(vp) === axes(xp)
    (; block_dims, cumulative_npoints_per_block, pointperm, buffers, indices,) = blocks
    Ms = map(Kernels.half_support, gs)
    Nt = length(buffers)  # usually equal to the number of threads
    nblocks = length(indices)
    Base.require_one_based_indexing(buffers)
    Base.require_one_based_indexing(indices)
    Threads.@threads :static for i ∈ 1:Nt
        j_start = (i - 1) * nblocks ÷ Nt + 1
        j_end = i * nblocks ÷ Nt
        @inbounds for j ∈ j_start:j_end
            block = buffers[i]
            I₀ = indices[j]

            # Indices of current block including padding
            inds_block = (I₀ + oneunit(I₀) - CartesianIndex(Ms)):(I₀ + CartesianIndex(block_dims) + CartesianIndex(Ms))
            copy_to_block!(block, us, inds_block)  # copy local data to `block` array

            # Iterate over all points in the current block
            a = cumulative_npoints_per_block[j] + 1
            b = cumulative_npoints_per_block[j + 1]
            for k ∈ a:b
                l = pointperm[k]
                # @assert blocks.blockidx[l] == j  # check that point is really in the current block
                x⃗ = xp[l]  # if points have not been permuted
                # x⃗ = xp[k]  # if points have been permuted (may be slightly faster here, but requires permutation in sort_points!)
                vp[l] = interpolate_blocked(gs, block, x⃗, Tuple(I₀))
            end
        end
    end
    vp
end

function copy_to_block!(block::AbstractArray, us::AbstractArray, inds::CartesianIndices)
    @assert size(block) == size(inds)
    Base.require_one_based_indexing(us)
    Ñs = size(us)
    @inbounds for i ∈ eachindex(block, inds)
        I = inds[i]
        is = map(wrap_periodic, Tuple(I), Ñs)  # wrap_periodic is defined in spreading.jl
        block[i] = us[is...]
    end
    us
end
