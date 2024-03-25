# Threaded permsort! implementation based on quicksort, adapted from the parallel sort!
# implementation by Nikos Pitsianis in
# https://discourse.julialang.org/t/parallel-quicksort-openmp/88499/18.
#
# This is used when applying blocking (see set_points!(::BlockData, ...)).

using Bumper: @no_escape, @alloc

const QUICKSORT_SEQ_THRESH = 1 << 9

@inline function partition_perm!(ix, A, pivot, left, right)
    @inbounds while left ≤ right
        while A[left] < pivot
            left += 1
        end
        while A[right] > pivot
            right -= 1
        end
        if left ≤ right
            A[left], A[right] = A[right], A[left]
            ix[left], ix[right] = ix[right], ix[left]
            left += 1
            right -= 1
        end
    end
    (left, right)
end

@inline function quicksort_perm!(
        ix, A, lo = firstindex(A), hi = lastindex(A);
        nthreads = Threads.nthreads(),
    )
    len = hi - lo + 1
    @inbounds if len ≤ 0
        return ix
    elseif len ≤ 32
        # Use sequential sortperm! from Base.
        ix_local = @view ix[lo:hi]
        A_local = @view A[lo:hi]
        quicksort_perm_base_case!(ix_local, A_local)
    else
        pivot = A[(lo + hi) >>> 1]
        left, right = partition_perm!(ix, A, pivot, lo, hi)
        if nthreads == 1 || len < QUICKSORT_SEQ_THRESH
            quicksort_perm!(ix, A, lo, right)
            quicksort_perm!(ix, A, left, hi)
        else
            t = Threads.@spawn quicksort_perm!(ix, A, lo, right)
            quicksort_perm!(ix, A, left, hi)
            wait(t)
        end
    end
    ix
end

@inline function quicksort_perm_base_case!(ix, A)
    Base.require_one_based_indexing(ix)
    @no_escape begin
        I = eltype(ix)
        perm = @alloc(I, length(ix))
        ix_sorted = @alloc(I, length(ix))
        sortperm!(perm, A; alg = QuickSort)
        @inbounds for j ∈ eachindex(perm)
            ix_sorted[j] = ix[perm[j]]
        end
        @inbounds for j ∈ eachindex(perm)
            ix[j] = ix_sorted[j]
        end
    end
    ix
end
