using StaticArrays: MVector

# Interpolate onto a single point
@kernel function interpolate_to_point_naive_kernel!(
        vp::NTuple{C},
        @Const(points::NTuple{D}),
        @Const(us::NTuple{C}),
        @Const(Δxs::NTuple{D}),           # grid step in each direction (oversampled grid)
        evaluate::NTuple{D, <:Function},  # can't be marked Const for some reason
        to_indices::NTuple{D, <:Function},
    ) where {C, D}
    i = @index(Global, Linear)
    x⃗ = map(xs -> @inbounds(xs[i]), points)

    # Determine grid dimensions.
    # Unlike in spreading, here `us` can be made of arrays of complex numbers, because we
    # don't perform atomic operations. This is why the code is simpler here.
    Ns = size(first(us))  # grid dimensions

    # Evaluate 1D kernels.
    gs_eval = map((f, x) -> f(x), evaluate, x⃗)

    # Determine indices to load from `u` arrays.
    inds = map(to_indices, gs_eval, Ns) do f, gdata, N
        f(gdata.i, N)
    end

    vals = map(gs_eval, Δxs) do geval, Δx
        geval.values .* Δx
    end

    # v⃗ = @inline interpolate_from_arrays(us, inds, vals)
    v⃗ = interpolate_from_arrays_gpu(us, inds, vals)

    for (dst, v) ∈ zip(vp, v⃗)
        @inbounds dst[i] = v
    end

    nothing
end

function interpolate!(
        backend::GPU,
        bd::Union{BlockDataGPU, NullBlockData},
        gs,
        vp_all::NTuple{C, AbstractVector},
        us::NTuple{C, AbstractArray},
        x⃗s::AbstractVector,
    ) where {C}
    # Note: the dimensions of arrays have already been checked via check_nufft_nonuniform_data.
    Base.require_one_based_indexing(x⃗s)  # this is to make sure that all indices match
    foreach(Base.require_one_based_indexing, vp_all)

    evaluate = map(Kernels.evaluate_kernel_func, gs)   # kernel evaluation functions
    to_indices = map(Kernels.kernel_indices_func, gs)  # functions returning spreading indices
    xs_comp = StructArrays.components(x⃗s)
    Δxs = map(Kernels.gridstep, gs)

    # TODO: use dynamically sized kernel? (to avoid recompilation, since number of points may change from one call to another)
    ndrange = size(x⃗s)  # iterate through points
    workgroupsize = default_workgroupsize(backend, ndrange)
    kernel! = interpolate_to_point_naive_kernel!(backend, workgroupsize, ndrange)
    kernel!(vp_all, xs_comp, us, Δxs, evaluate, to_indices)

    vp_all
end

@inline function interpolate_from_arrays_gpu(
        us::NTuple{C, AbstractArray{T, D}},
        inds_mapping::NTuple{D, Tuple},
        vals::NTuple{D, NTuple{M, Tg}},
    ) where {T, C, D, M, Tg <: AbstractFloat}
    if @generated
        gprod_init = Symbol(:gprod_, D + 1)  # the name of this variable is important!
        quote
            inds = map(eachindex, vals)
            vs = zero(MVector{$C, $T})
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
                    for n ∈ 1:$C
                        @inbounds vs[n] += gprod_1 * us[n][js...]
                    end
                end
            )
            Tuple(vs)
        end
    else
        # Note: the trick of splitting the first dimension from the other ones really helps with
        # performance on the GPU.
        vs = zero(MVector{C, T})
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
                for n ∈ eachindex(vs)
                    vs[n] += gprod * us[n][i, is_tail...]
                end
            end
        end
        Tuple(vs)
    end
end
