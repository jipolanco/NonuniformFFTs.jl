# Code adapted from the BijectiveHilbert.jl package by Andrew Dolgert (MIT license).
#
# Hilbert sorting is a method for sorting a set of points in n-dimensional space so that
# points which are physically close are also close in memory. See for instance:
#
# - https://en.wikipedia.org/wiki/Hilbert_curve
# - https://charlesreid1.com/wiki/Hilbert_Sort
# - https://doc.cgal.org/latest/Spatial_sorting/index.html
# - https://computingkitchen.com/BijectiveHilbert.jl/stable/hilbert/
#
# We define a GlobalGrayStatic{T,B} type adapted from the BijectiveHilbert.GlobalGray{T}
# type. The differences are that GlobalGrayStatic contains the number of bits `B` as a
# static parameter, and that it simply does not include the number
# of dimensions `n`. Instead, `n` is inferred from the input to `encode_hilbert_zero`, which
# should be a (statically-sized) `Tuple` instead of a `Vector` as in BijectiveHilbert.jl.
#
# Moreover, functions for computing the Hilbert index have been adapted for working with
# tuples and have no allocations. The performance is greatly improved compared to the
# original GlobalGray type. Our code can also be called from GPU kernels, since it only uses
# tuples and MVectors.
#
# We only implement encoding as we currently don't need decoding.
#
# See also https://computingkitchen.com/BijectiveHilbert.jl/stable/globalgray/#Global-Gray
# for some details on the algorithm.

abstract type HilbertSortingAlgorithm{T, B} end

Base.eltype(::HilbertSortingAlgorithm{T}) where {T} = T
nbits(::HilbertSortingAlgorithm{T, B}) where {T, B} = B

"""
    GlobalGrayStatic(T <: Unsigned, B::Int) <: HilbertSortingAlgorithm{T, B}
    GlobalGrayStatic(B::Int, N::Int)

Hilbert sorting algorithm adapted from the `GlobalGray` algorithm in BijectiveHilbert.jl.

Here `T <: Unsigned` is the type of the Hilbert index and `B` is the required number of
bits per dimension. For `B` bits, the algorithm creates a grid of size ``2^B`` in each
direction.

The second constructor chooses the smallest unsigned type `T` based on the required number
of bits `B` and the space dimension `N`. If they are constants, they will likely be
constant-propagated so that `T` is inferred by the compiler.

The total number of bits required is `nbits = B * N`. For instance, if `N = 3` and `B = 4`
(corresponding to a grid of dimensions ``(2^4)^3 = 16^3``), then `nbits = 12` and the
smallest possible type `T` is `UInt16`, whose size is `8 * sizeof(UInt16) = 16` bits (as its
name suggests!).
"""
struct GlobalGrayStatic{T, B} <: HilbertSortingAlgorithm{T, B} end

GlobalGrayStatic(::Type{T}, B::Int) where {T} = GlobalGrayStatic{T, B}()

Base.@constprop :aggressive function GlobalGrayStatic(B::Int, N::Int)
    nbits = B * N  # minimal number of bits needed
    T = large_enough_unsigned_static(Val(nbits))
    GlobalGrayStatic{T, B}()
end

@inline function large_enough_unsigned_static(::Val{nbits}) where {nbits}
    if @generated
        unsigned_types = (UInt8, UInt16, UInt32, UInt64, UInt128)
        for T ∈ unsigned_types
            if 8 * sizeof(T) ≥ nbits
                return :($T)
            end
        end
        :(nothing)
    else
        unsigned_types = (UInt8, UInt16, UInt32, UInt64, UInt128)
        for T ∈ unsigned_types
            if 8 * sizeof(T) ≥ nbits
                return T
            end
        end
        nothing
    end
end

"""
    encode_hilbert_zero(algorithm::GlobalGrayStatic{T}, X::NTuple{N, <:Integer}) -> T

Return Hilbert index associated to location `X` in `N`-dimensional space.

Values in `X` usually correspond to indices on a grid. This function takes indices which start at 0 (!!).
"""
function encode_hilbert_zero(::GlobalGrayStatic{T, B}, X::NTuple) where {T, B}
    X = axes_to_transpose(X, Val(B))
    interleave_transpose(T, X, Val(B))::T
end

function axes_to_transpose(X_in::NTuple{N,T}, ::Val{b}) where {N, b, T <: Integer}
    X = MVector(X_in)
    M = one(T) << (b - 1)
    # Inverse undo
    Q = M
    @inbounds while Q > one(T)
        P = Q - one(T)
        for io ∈ 1:N
            if !iszero(X[io] & Q)
                X[1] ⊻= P
            else
                t = (X[1] ⊻ X[io]) & P
                X[1] ⊻= t
                X[io] ⊻= t
            end
        end
        Q >>= one(Q)
    end
    # Gray encode
    for jo ∈ 2:N
        @inbounds X[jo] ⊻= X[jo - 1]
    end
    t2 = zero(T)
    Q = M
    @inbounds while Q > one(T)
        if !iszero(X[N] & Q)
            t2 ⊻= (Q - one(T))
        end
        Q >>= one(T)
    end
    for ko ∈ eachindex(X)
        @inbounds X[ko] ⊻= t2
    end
    Tuple(X)
end

function interleave_transpose(::Type{T}, x::NTuple, ::Val{b}) where {T, b}
    N = length(x)
    S = eltype(x)
    h = zero(T)
    @inbounds for i ∈ 0:(b - 1)
        for d ∈ eachindex(x)
            ith_bit = Bool((x[d] & (one(S) << i)) >> i)  # 0 or 1
            h |= T(ith_bit << (i * N + N - d))
        end
    end
    h
end
