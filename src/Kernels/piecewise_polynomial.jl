# Approximate kernel function using piecewise polynomials.
# On each subinterval of size 1/M (where M is the integer kernel half-width), we approximate
# the kernel using a polynomial of degree N - 1.
# Note that the piecewise polynomial is allowed (in principle) to be discontinuous between
# two subintervals.
#
# The idea is heavily inspired by FINUFFT (see Barnett, Magland & af Klinteberg, SIAM J.
# Sci. Comput. 2019), with a few small differences:
#
# - we don't need to generate specific code for specific oversampling factors σ or
#   half-widths M. Our implementation is generic and even then it's extremely efficient;
#
# - we interpolate from function values at Chebyshev nodes, instead of evaluating the
#   function in the complex plane. This is inspired by a paper by Shamshigar, Bagge &
#   Tornberg (J. Chem. Phys. 2021).

using StaticArrays: SVector

function solve_polynomial_coefficients(f::F, ::Type{T}, ::Val{N}) where {F, T, N}
    # Note: we could use a MMatrix{N,N,T} to avoid some allocations, but that greatly
    # increases the compilation time...
    A = Matrix{T}(undef, N, N)
    xs = SVector(ntuple(i -> cospi(T(i - 1/2) / N), Val(N)))  # interpolate at Chebyshev points
    ys = map(f, xs)
    xs_pow = zero(xs) .+ true  # = 1
    for j ∈ 1:N
        A[:, j] = xs_pow
        xs_pow = map(*, xs_pow, xs)  # = xs_pow .* xs
    end
    NTuple{N}(A \ ys)
end

function transpose_tuple_of_tuples(cs::NTuple{L, NTuple{N}}) where {L, N}
    ntuple(Val(N)) do j
        ntuple(i -> cs[i][j], Val(L))
    end
end

# L = 2M: total number of subintervals in [-1, 1].
function solve_piecewise_polynomial_coefficients(
        f::F, ::Type{T}, ::Val{M}, ::Val{N},
    ) where {F, T, M, N}
    L = 2M
    cs = ntuple(Val(L)) do j
        # Note: we go from right (+1) to left (-1).
        h = 1 - 2 * (j - 1/2) / L  # midpoint of interval
        δ = 1 / L                  # half-width of interval
        solve_polynomial_coefficients(T, Val(N)) do x
            y = h + x * δ  # Transform x ∈ [-1, 1] → y ∈ [h - δ, h + δ]
            f(y)
        end
    end
    transpose_tuple_of_tuples(cs)
end

function evaluate_piecewise(δ, cs::NTuple)
    L = length(first(cs))
    # @assert 0 ≤ δ < 2/L == 1/M
    x = L * δ - 1  # in [-1, 1]
    evaluate_horner(x, cs)
end

# Evaluate multiple polynomials at the same location `x` using Horner's method.
# This is very fast due to SIMD vectorisation.
function evaluate_horner(x, cs::NTuple)
    ys = @inbounds SVector(cs[end])
    js = @inbounds reverse(eachindex(cs)[1:end - 1])
    for j ∈ js
        cj = @inbounds SVector(cs[j])
        ys = muladd(x, ys, cj)
    end
    Tuple(ys)
end
