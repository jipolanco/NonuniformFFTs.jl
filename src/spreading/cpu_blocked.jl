using Atomix: Atomix

function split_periodic(
        Ia::CartesianIndex{D}, Ib::CartesianIndex{D}, Ns::Dims{D},
    ) where {D}
    ntuple(Val(D)) do j
        split_periodic(Ia[j]:Ib[j], Ns[j])
    end
end

# Split range into two contiguous ranges in 1:N after periodic wrapping.
# We assume the input range only goes outside 1:N either on the left or the right of the
# range (and by less than N values).
# This requires the block size B to be smaller than the dataset size N.
# More exactly, we need B ≤ N - M where M is the half kernel support (= the block padding).
function split_periodic(irange::AbstractUnitRange, N)
    T = typeof(irange)
    if irange[begin] < 1
        # We assume the range includes the 1.
        n = searchsortedlast(irange, 1)
        # @assert n > firstindex(irange) && irange[n] == 1
        a = (irange[begin]:irange[n - 1]) .+ N  # indices [..., N - 1, N]
        b = irange[n]:irange[end]  # indices [1, 2, ...]
    elseif last(irange) > N
        # We assume the range includes N.
        n = searchsortedlast(irange, N)
        # @assert n < lastindex(irange) && irange[n] == N
        a = irange[begin]:irange[n]
        b = (irange[n + 1]:irange[end]) .- N
    else
        # A single contiguous range, plus an empty one.
        a = irange
        b = T(0:-1)  # empty range
    end
    (a, b) :: Tuple{T, T}
end

function spread_from_point_blocked!(
        gs::NTuple{D, AbstractKernelData},
        evalmode::EvaluationMode,
        us::NTuple{C, AbstractArray{T, D}} where {T},
        x⃗₀::NTuple{D, Number},
        vs::NTuple{C, Number},
        I₀::NTuple,
    ) where {C, D}
    # Evaluate 1D kernels.
    gs_eval = map((g, x) -> Kernels.evaluate_kernel(evalmode, g, x), gs, x⃗₀)

    Ms = map(Kernels.half_support, gs)
    δs = Ms .- I₀  # index offset

    # Determine indices to write in `u` arrays.
    inds = map(gs_eval, gs, δs) do gdata, g, δ
        is = Kernels.kernel_indices(gdata.i, g)  # note: this variant doesn't perform periodic wrapping
        is .+ δ  # shift to beginning of current block
    end
    Is = CartesianIndices(inds)
    # checkbounds.(us, Ref(Is))  # check that indices fall inside the output array

    vals = map(g -> g.values, gs_eval)
    spread_onto_arrays_blocked!(us, Is, vals, vs)

    us
end

# We find it's faster to use a low-level call to memset (as opposed to a `for` loop, or
# `fill!`). In fact, this mostly seems to be the case for complex data, while for real data
# using `fill!` gives the same performance...
# Using memset only makes sense if the arrays are contiguous in memory (DenseArray).
function fill_with_zeros_serial!(us_all::NTuple{C, A}) where {C, A <: DenseArray}
    # We assume all arrays in the tuple have the same type and shape.
    inds = eachindex(first(us_all)) :: AbstractVector  # make sure array uses linear indices
    @assert isone(first(inds))  # 1-based indexing
    @assert all(us -> eachindex(us) === inds, us_all)  # all arrays have the same indices
    GC.@preserve us_all begin
        for us ∈ us_all
            p = pointer(us)
            n = length(us) * sizeof(eltype(us))
            val = zero(Cint)
            ccall(:memset, Ptr{Cvoid}, (Ptr{Cvoid}, Cint, Csize_t), p, val, n)
        end
    end
    us_all
end

to_real_array_cpu(u::Array{<:Real}) = u

function to_real_array_cpu(u::Array{Complex{T}}) where {T <: Real}
    # reinterpret(reshape, T, u)  # this is inefficient!
    p = convert(Ptr{T}, pointer(u))
    unsafe_wrap(Array, p, (2, size(u)...))::Array{T}
end

function spread_from_points!(
        ::CPU,
        callback::Callback,
        transform_fold::F,
        bd::BlockDataCPU,
        gs,
        evalmode::EvaluationMode,
        us_all::NTuple{C, AbstractArray},  # we assume this is completely set to zero
        xp::NTuple{D, AbstractVector},
        vp_all::NTuple{C, AbstractVector};
        cpu_use_atomics::Bool = false,
    ) where {F <: Function, Callback <: Function, C, D}
    (; block_dims, pointperm, indices,) = bd
    Z = eltype(eltype(us_all))
    Ms = map(Kernels.half_support, gs)
    block_dims_padded = @. block_dims + 2 * Ms
    Base.require_one_based_indexing(indices)
    lck = ReentrantLock()

    block_inds = eachindex(IndexLinear(), indices)  # block indices (= 1:nblocks)

    # Reinterpret output array as real-valued array if it's complex.
    # This is needed if we're using atomics on complex data.
    us_real = map(to_real_array_cpu, us_all)

    scheduler = DynamicScheduler(chunking = false)  # disable chunking to improve load balancing
    tforeach(block_inds; scheduler) do block_idx  # iterate over blocks
        @inline
        buf = Bumper.default_buffer()  # task-local buffer
        @no_escape buf begin
            block = ntuple(Val(C)) do component
                @alloc(Z, block_dims_padded...)
            end
            a = @inbounds bd.cumulative_npoints_per_block[block_idx]
            b = @inbounds bd.cumulative_npoints_per_block[block_idx + 1]
            if b > a  # if there are points in this block (otherwise there's nothing to do)
                # Iterate over all points in the current block
                @inbounds I₀ = indices[block_idx]
                fill_with_zeros_serial!(block)
                for i ∈ (a + 1):b
                    @inbounds l = pointperm[i]
                    j = if bd.sort_points === True()
                        i  # if points have been permuted (may be slightly faster here, but requires permutation in set_points!)
                    else
                        l  # if points have not been permuted
                    end
                    x⃗ = map(xp -> transform_fold(@inbounds(xp[j])), xp)
                    vs = map(vp -> @inbounds(vp[l]), vp_all)  # values at the non-uniform point x⃗
                    vs_new = @inline callback(vs, j)
                    spread_from_point_blocked!(gs, evalmode, block, x⃗, vs_new, Tuple(I₀))
                end

                # Indices of current block including padding
                Ia = I₀ + oneunit(I₀) - CartesianIndex(Ms)
                Ib = I₀ + CartesianIndex(block_dims) + CartesianIndex(Ms)
                inds_split = split_periodic(Ia, Ib, size(first(us_all)))

                # Add data from block to output array.
                if cpu_use_atomics
                    add_from_block!(us_real, block, inds_split; atomics = Val(true))
                else
                    # This is executed by only one thread at a time (can be slower for many threads)
                    lock(lck) do
                        add_from_block!(us_real, block, inds_split; atomics = Val(false))
                    end
                end
            end  # b > a
        end  # @no_escape
    end  # tforeach

    us_all
end

function add_from_block!(
        us_all::NTuple{C, AbstractArray},
        block::NTuple{C, AbstractArray},
        inds_wrapped::NTuple;
        kws...,
    ) where {C}
    for i ∈ 1:C
        _add_from_block!(us_all[i], block[i], inds_wrapped; kws...)
    end
    us_all
end

# Recursively generate a loop over `d` dimensions, where each dimension is split into 2 sets
# of indices.
# This function generates a Julia expression which is included in a @generated function.
function _generate_split_loop_expr(d, inds, loop_core)
    if d == 0
        return loop_core
    end
    jd = Symbol(:j_, d)  # e.g. j_3
    ex_prev = _generate_split_loop_expr(d - 1, inds, loop_core)
    quote
        for $jd ∈ $inds[$d][1]
            $ex_prev
        end
        for $jd ∈ $inds[$d][2]
            $ex_prev
        end
    end
end

@inline function cpu_add_atomic!(us::AbstractArray{<:Real}, js::Tuple, w::Real)
    @inbounds begin
        Atomix.@atomic :monotonic us[js...] += w
    end
    nothing
end

@inline function cpu_add_atomic!(us::AbstractArray{<:Real}, js::Tuple, w::Complex)
    @inbounds begin
        Atomix.@atomic :monotonic us[1, js...] += real(w)
        Atomix.@atomic :monotonic us[2, js...] += imag(w)
    end
    nothing
end

function _add_from_block!(
        us::AbstractArray{T},
        ws::AbstractArray{Z, D},
        inds_wrapped::NTuple{D, NTuple{2, UnitRange}};
        atomics::Val,
    ) where {T, Z, D}
    if @generated
        loop_core = quote
            n += 1
            js = @ntuple $D j
            if atomics === Val(true)
                cpu_add_atomic!(us, js, ws[n])  # us[j_1, j_2, ..., j_D] += ws[n]
            elseif Z <: Complex
                us[1, js...] += real(ws[n])  # us[j_1, j_2, ..., j_D] += ws[n]
                us[2, js...] += imag(ws[n])  # us[j_1, j_2, ..., j_D] += ws[n]
            else
                us[js...] += ws[n]  # us[j_1, j_2, ..., j_D] += ws[n]
            end
        end
        ex_loop = _generate_split_loop_expr(D, :inds_wrapped, loop_core)
        quote
            number_of_indices_per_dimension = @ntuple($D, i -> sum(length, inds_wrapped[i]))
            # @assert size(ws) == number_of_indices_per_dimension
            Base.require_one_based_indexing(ws)
            Base.require_one_based_indexing(us)
            n = 0
            @inbounds begin
                $ex_loop
            end
            # @assert n == length(ws)
            us
        end
    else
        # @assert size(ws) == map(tup -> sum(length, tup), inds_wrapped)
        Base.require_one_based_indexing(ws)
        Base.require_one_based_indexing(us)
        iters = map(enumerate ∘ Iterators.flatten, inds_wrapped)
        iter_first, iters_tail =  first(iters), Base.tail(iters)
        @inbounds for inds_tail ∈ Iterators.product(iters_tail...)
            is_tail = map(first, inds_tail)
            js_tail = map(last, inds_tail)
            for (i, j) ∈ iter_first
                if atomics === Val(true)
                    cpu_add_atomic!(us, (j, js_tail...), ws[i, is_tail...])
                else
                    us[j, js_tail...] += ws[i, is_tail...]
                end
            end
        end
        us
    end
end

# This is basically the same as the non-blocked version, but uses CartesianIndices instead
# of tuples (since indices don't "jump" due to periodic wrapping).
function spread_onto_arrays_blocked!(
        us::NTuple{C, AbstractArray{T, D}},
        Is::CartesianIndices{D},
        vals::NTuple{D, Tuple},
        vs::NTuple{C, T},
    ) where {C, T, D}
    # NOTE: When C > 1, we found that we gain nothing (in terms of performance) by combining
    # operations over C arrays at once. Things actually get much slower for some reason.
    # So we simply perform the same operation C times.
    for i ∈ 1:C
        _spread_onto_arrays_blocked!(us[i], Is, vals, vs[i])
    end
    us
end

function _spread_onto_arrays_blocked!(
        u::AbstractArray{T, D},
        Is::CartesianIndices{D},
        vals::NTuple{D, Tuple},
        v::T,
    ) where {T, D}
    if @generated
        gprod_init = Symbol(:gprod_, D)  # the name of this variable is important!
        quote
            inds = map(eachindex, vals)
            $gprod_init = v
            @inbounds @nloops(
                $(D - 1),
                i,
                d -> inds[d + 1],  # for i_d ∈ inds[d + 1]
                d -> begin
                    gprod_d = gprod_{d + 1} * vals[d + 1][i_d]  # add factor for dimension d + 1
                end,
                begin
                    is_tail = @ntuple $(D - 1) i
                    I₀ = Is[inds[1][1], is_tail...]
                    n = LinearIndices(u)[I₀]
                    for i_0 ∈ inds[1]
                        gprod_0 = gprod_1 * vals[1][i_0]
                        u[n] += gprod_0
                        n += 1
                    end
                end,
            )
            u
        end
    else
        inds = map(eachindex, vals)
        Js = CartesianIndices(inds)
        @inbounds for J ∈ Js
            gs = map(inbounds_getindex, vals, Tuple(J))
            gprod = v * prod(gs)
            I = Is[J]
            u[I] += gprod
        end
        u
    end
end
