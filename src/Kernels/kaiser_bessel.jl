export KaiserBesselKernel

using Bessels: besseli0, besseli1

# This is the equivalent variance of a KB window with width w = 1.
# Should be compared to the variance of a Gaussian window.
kb_equivalent_variance(β) = besseli0(β) / (β * besseli1(β))

struct KaiserBesselKernel <: AbstractKernel end

"""
    KaiserBesselKernelData(HalfSupport(M), Δx, β)

Create a [Kaiser–Bessel](https://en.wikipedia.org/wiki/Kaiser_window#Definition) kernel.

The kernel parameters are:

- `M`: the half-support of the kernel (an integer value);
- `Δx`: the spacing of the (oversampled) grid;
- `β`: the Kaiser–Bessel shape parameter.

Here the half-kernel size in "physical" units is ``w = M * Δx``.
This means that if we want to evaluate the kernel around a source location ``x``,
then the evaluation points will be in ``[x - w, x + w)``.

More precisely, the evaluation points will be ``x_i = x - δ + i * Δx``, where the point
``x_0 = x - δ`` is the nearest point to the left of ``x`` which is on the grid
(that is, ``0 ≤ δ < Δx``), and ``i ∈ \\left{-M, -M + 1, …, M - 2, M - 1 \\right}``.

## Optimal shape parameter β

For the purposes of computing NUFFTs, the optimal shape parameter is

```math
β = M π \\left( 2 - \\frac{1}{σ} \\right)
```

where ``σ ≥ 1`` is the NUFFT oversampling parameter.
See for instance Potts & Steidl, SIAM J. Sci. Comput. 2003, eq. (5.12).
In other words, FFTs must be performed over a grid of size ``Ñ = σN`` where ``N`` is the
size of the grid of interest.
"""
struct KaiserBesselKernelData{
        M, T <: AbstractFloat, ApproxCoefs <: AbstractArray{T},
    } <: AbstractKernelData{KaiserBesselKernel, M, T}
    Δx :: T  # grid spacing
    σ  :: T  # equivalent kernel width (for comparison with Gaussian)
    w  :: T  # actual kernel half-width (= M * Δx)
    β  :: T  # KB parameter
    β² :: T
    I₀_at_β :: T
    cs :: ApproxCoefs  # coefficients of polynomial approximation
    gk :: Vector{T}

    function KaiserBesselKernelData{M}(Δx::T, β::T) where {M, T <: AbstractFloat}
        w = M * Δx
        σ = sqrt(kb_equivalent_variance(β)) * w
        β² = β * β
        I₀_at_β = besseli0(β)
        gk = Vector{T}(undef, 0)
        Npoly = M + 4  # degree of polynomial is d = Npoly - 1
        cs = solve_piecewise_polynomial_coefficients(T, Val(M), Val(Npoly)) do x
            besseli0(β * sqrt(1 - x^2)) / I₀_at_β
        end
        new{M, T, typeof(cs)}(Δx, σ, w, β, β², I₀_at_β, cs, gk)
    end
end

KaiserBesselKernelData(::HalfSupport{M}, args...) where {M} =
    KaiserBesselKernelData{M}(args...)

function optimal_kernel(::KaiserBesselKernel, h::HalfSupport{M}, Δx, σ) where {M}
    # Set the optimal kernel shape parameter given the wanted support M and the oversampling
    # factor σ. See Potts & Steidl 2003, eq. (5.12).
    γ = 0.980  # empirical "safety factor" which slightly improves accuracy, as in FINUFFT (where γ = 0.976)
    β = oftype(Δx, M * π * (2 - 1 / σ)) * γ
    KaiserBesselKernelData(h, Δx, β)
end

function evaluate_fourier(g::KaiserBesselKernelData, k::Number)
    (; β², w, I₀_at_β,) = g
    q = w * k
    s = sqrt(β² - q^2)  # this is always real (assuming β ≥ Mπ)
    2 * w * sinh(s) / (I₀_at_β * s)
end

function evaluate_kernel(g::KaiserBesselKernelData{M}, x, i::Integer) where {M}
    # Evaluate in-between grid points xs[(i - M):(i + M)].
    # Note: xs[j] = (j - 1) * Δx
    (; w,) = g
    X = x / w - (i - 1) / M  # source position relative to xs[i]
    # @assert 0 ≤ X < 1 / M
    values = evaluate_piecewise(X, g.cs)
    (; i, values,)
end
