function interpolate_blocked(
        gs::NTuple{D},
        evalmode::EvaluationMode,
        us::NTuple{C, AbstractArray{T, D}} where {T},
        x⃗::NTuple{D},
        I₀::NTuple{D},
    ) where {D, C}
    @assert C > 0
    map(Base.require_one_based_indexing, us)
    Ns = size(first(us))
    @assert all(u -> size(u) === Ns, us)

    # Evaluate 1D kernels.
    gs_eval = map((g, x) -> Kernels.evaluate_kernel(evalmode, g, x), gs, x⃗)

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

# This is equivalent to spread_onto_arrays_blocked! in spreading.
function interpolate_from_arrays_blocked(
        us::NTuple{C, AbstractArray{T, D}},  # `T` can be complex
        Is::CartesianIndices{D},
        vals::NTuple{D, NTuple{M, Tg}},       # `Tg` is always real
    ) where {C, D, M, T, Tg <: AbstractFloat}
    if @generated
        gprod_init = Symbol(:gprod_, D)  # the name of this variable is important!
        quote
            inds = map(eachindex, vals)
            @nexprs $C j -> (v_j = zero($T))
            @nextract $C u us  # creates variables u_1, u_2, ..., u_C
            $gprod_init = one($Tg)
            # Split loop onto dimensions 1 and 2:D.
            # The loop over the first (fastest) dimension is avoided and SVectors are used
            # instead.
            @inbounds @nloops(
                $(D - 1),
                i,
                d -> inds[d + 1],  # for i_d ∈ inds[d]
                d -> begin
                    gprod_d = gprod_{d + 1} * vals[d + 1][i_d]  # add factor for dimension d + 1
                end,
                begin
                    is_tail = @ntuple $(D - 1) i
                    # Try to automatically take advantage of SIMD vectorisation
                    # (effectiveness may depend on half support size M...).
                    gs = gprod_1 * SVector{$M}(vals[1])
                    @nexprs $C j -> begin
                        udata_j = @ntuple $M k -> u_j[Is[k, is_tail...]]
                        uvec_j = SVector{$M}(udata_j)
                        v_j += gs ⋅ uvec_j
                    end
                end,
            )
            @ntuple $C v
        end
    else
        vs = ntuple(_ -> zero(eltype(first(us))), Val(C))
        inds = map(eachindex, vals)
        inds_tail = Base.tail(inds)
        vals_tail = Base.tail(vals)
        @inbounds for J_tail ∈ CartesianIndices(inds_tail)
            js_tail = Tuple(J_tail)
            gs_tail = map(inbounds_getindex, vals_tail, js_tail)
            gprod_tail = prod(gs_tail)
            gs = gprod_tail * SVector(vals[1])
            vs_new = map(us) do u
                @inline
                udata = ntuple(k -> @inbounds(u[Is[k, js_tail...]]), Val(M))
                gs ⋅ SVector(udata)
            end
            vs = vs .+ vs_new
        end
        vs
    end
end

function interpolate!(
        backend::CPU,
        bd::BlockData,
        gs,
        evalmode::EvaluationMode,
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
        @inbounds for j ∈ j_start:j_end
            a = bd.cumulative_npoints_per_block[j]
            b = bd.cumulative_npoints_per_block[j + 1]
            a == b && continue  # no points in this block (otherwise b > a)

            # Indices of current block including padding
            I₀ = indices[j]
            Ia = I₀ + oneunit(I₀) - CartesianIndex(Ms)
            Ib = I₀ + CartesianIndex(block_dims) + CartesianIndex(Ms)
            inds_split = split_periodic(Ia, Ib, size(first(us)))
            copy_to_block!(block, us, inds_split)

            # Iterate over all points in the current block
            for k ∈ (a + 1):b
                l = pointperm[k]
                # @assert bd.blockidx[l] == j  # check that point is really in the current block
                if bd.sort_points === True()
                    x⃗ = x⃗s[k]  # if points have been permuted (may be slightly faster here, but requires permutation in set_points!)
                else
                    x⃗ = x⃗s[l]  # if points have not been permuted
                end
                vs = interpolate_blocked(gs, evalmode, block, x⃗, Tuple(I₀)) :: NTuple{C}  # non-uniform values at point x⃗
                for (vp, v) ∈ zip(vp_all, vs)
                    @inbounds vp[l] = v
                end
            end
        end
    end

    vp_all
end

# This is equivalent to add_from_block! in spreading.
function copy_to_block!(
        block::NTuple{C, AbstractArray},
        us_all::NTuple{C, AbstractArray},
        inds_wrapped::NTuple{D, NTuple{2, UnitRange}},
    ) where {C, D}
    for i ∈ 1:C
        _copy_to_block!(block[i], us_all[i], inds_wrapped)
    end
    block
end

function _copy_to_block!(
        ws::AbstractArray{T, D},
        us::AbstractArray{T, D},
        inds_wrapped::NTuple{D, NTuple{2, UnitRange}},
    ) where {T, D}
    if @generated
        loop_core = quote
            n += 1
            js = @ntuple $D j
            ws[n] = us[js...]
        end
        ex_loop = _generate_split_loop_expr(D, :inds_wrapped, loop_core)
        quote
            number_of_indices_per_dimension = @ntuple($D, i -> sum(length, inds_wrapped[i]))
            @assert size(ws) == number_of_indices_per_dimension
            Base.require_one_based_indexing(ws)
            Base.require_one_based_indexing(us)
            n = 0
            @inbounds begin
                $ex_loop
            end
            @assert n == length(ws)
            ws
        end
    else
        @assert size(ws) == map(tup -> sum(length, tup), inds_wrapped)
        Base.require_one_based_indexing(ws)
        Base.require_one_based_indexing(us)
        iters = map(enumerate ∘ Iterators.flatten, inds_wrapped)
        iter_first, iters_tail =  first(iters), Base.tail(iters)
        @inbounds for inds_tail ∈ Iterators.product(iters_tail...)
            is_tail = map(first, inds_tail)
            js_tail = map(last, inds_tail)
            for (i, j) ∈ iter_first
                ws[i, is_tail...] = us[j, js_tail...]
            end
        end
        ws
    end
end
