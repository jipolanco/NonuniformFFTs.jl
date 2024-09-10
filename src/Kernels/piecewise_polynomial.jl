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
using LinearAlgebra: lu!, ldiv!

# Note: we could use StaticArrays here but that will greatly increase the compilation time
# (unnecessarily, since this function is called just when creating a plan, and it's quite
# cheap).
function solve_polynomial_coefficients!(f::F, bufs) where {F}
    (; A, xs, xs_pow, ys,) = bufs
    N = length(xs)
    @assert size(A) == (N, N)
    @assert length(xs_pow) == N
    @assert length(ys) == N
    for i ∈ eachindex(xs, ys, xs_pow)
        ys[i] = f(xs[i])
        xs_pow[i] = 1
    end
    @inbounds for j ∈ 1:N
        for i ∈ 1:N
            A[i, j] = xs_pow[i]
            xs_pow[i] *= xs[i]
        end
    end
    Afact = lu!(A)
    ldiv!(Afact, ys)
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
    bufs = (
        A =  Matrix{T}(undef, N, N),
        xs = Vector{T}(undef, N),
        xs_pow = Vector{T}(undef, N),
        ys = Vector{T}(undef, N),
    )
    for i ∈ 1:N
        bufs.xs[i] = cospi(T(i - 1/2) / N)  # interpolate at Chebyshev points
    end
    cs = ntuple(Val(L)) do j
        # Note: we go from right (+1) to left (-1).
        h = 1 - 2 * (j - 1/2) / L  # midpoint of interval
        δ = 1 / L                  # half-width of interval
        cs_j = solve_polynomial_coefficients!(bufs) do x
            y = h + x * δ  # Transform x ∈ [-1, 1] → y ∈ [h - δ, h + δ]
            f(y)
        end
        ntuple(i -> cs_j[i], Val(N))
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
