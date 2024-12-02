export KaiserBesselKernel

using Bessels: Bessels

# This is the equivalent variance of a KB window with width w = 1.
# Should be compared to the variance of a Gaussian window.
kb_equivalent_variance(β) = Bessels.besseli0(β) / (β * Bessels.besseli1(β))

@doc raw"""
    KaiserBesselKernel <: AbstractKernel
    KaiserBesselKernel([β])

Represents a [Kaiser–Bessel](https://en.wikipedia.org/wiki/Kaiser_window#Definition)
spreading kernel.

# Definition

```math
ϕ(x) = I₀ \left(β \sqrt{1 - x²} \right)
\quad \text{ for } |x| ≤ 1
```

where ``I₀`` is the zeroth-order [modified Bessel
function](https://en.wikipedia.org/wiki/Modified_Bessel_function#Modified_Bessel_functions:_I%CE%B1,_K%CE%B1)
of the first kind and ``β`` is a shape factor.

# Fourier transform

```math
ϕ̂(k) = 2 \frac{\sinh\left( \sqrt{β² - k²} \right)}{\sqrt{β² - k²}}
```

# Parameter selection

By default, the shape parameter is chosen to be [1]

```math
β = γ M π \left( 2 - \frac{1}{σ} \right)
```

where ``M`` is the kernel half-width and ``σ`` the oversampling factor.
Moreover, ``γ = 0.980`` is an empirical "safety factor", similarly to the one used by
FINUFFT [2], which slightly improves accuracy.

This default value can be overriden by explicitly passing a ``β`` value.

""" *
"""
# Implementation details

Since the evaluation of Bessel functions can be costly, this kernel is efficiently evaluated
via an accurate piecewise polynomial approximation.
We use the same method originally proposed for FINUFFT [2] and later discussed by
Shamshirgar et al. [3].

[1] Potts & Steidl, SIAM J. Sci. Comput. **24**, 2013 (2003) \\
[2] Barnett, Magland & af Klinteberg, SIAM J. Sci. Comput. **41**, C479 (2019) \\
[3] Shamshirgar, Bagge & Tornberg, J. Chem. Phys. **154**, 164109 (2021)
"""
struct KaiserBesselKernel{Beta <: Union{Nothing, Real}} <: AbstractKernel
    β :: Beta
end

KaiserBesselKernel() = KaiserBesselKernel(nothing)

"""
    KaiserBesselKernelData(HalfSupport(M), Δx, β)

Create a [Kaiser–Bessel](https://en.wikipedia.org/wiki/Kaiser_window#Definition) kernel.

The kernel parameters are:

- `M`: the half-support of the kernel (an integer value);
- `Δx`: the spacing of the (oversampled) grid;
- `β`: the Kaiser–Bessel shape parameter.

Here the half-kernel size in "physical" units is ``w = M Δx``.
This means that if we want to evaluate the kernel around a source location ``x``,
then the evaluation points will be in ``[x - w, x + w)``.

More precisely, the evaluation points will be ``x_i = x - δ + i Δx``, where the point
``x_0 = x - δ`` is the nearest point to the left of ``x`` which is on the grid
(that is, ``0 ≤ δ < Δx``), and ``i ∈ \\left\\{-M, -M + 1, …, M - 2, M - 1 \\right\\}``.

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
        M, T <: AbstractFloat, ApproxCoefs <: NTuple,
        FourierCoefs <: AbstractVector{T},
    } <: AbstractKernelData{KaiserBesselKernel, M, T}
    Δx :: T  # grid spacing
    σ  :: T  # equivalent kernel width (for comparison with Gaussian)
    w  :: T  # actual kernel half-width (= M * Δx)
    β  :: T  # KB parameter
    β² :: T
    cs :: ApproxCoefs  # coefficients of polynomial approximation
    gk :: FourierCoefs

    function KaiserBesselKernelData{M}(Δx::T, σ::T, w::T, β::T, β²::T, cs, gk) where {M, T <: AbstractFloat}
        new{M, T, typeof(cs), typeof(gk)}(Δx, σ, w, β, β², cs, gk)
    end

    function KaiserBesselKernelData{M}(backend::KA.Backend, Δx::T, β::T) where {M, T <: AbstractFloat}
        w = M * Δx
        σ = sqrt(kb_equivalent_variance(β)) * w
        β² = β * β
        gk = KA.allocate(backend, T, 0)
        Npoly = M + 4  # degree of polynomial is d = Npoly - 1
        cs = solve_piecewise_polynomial_coefficients(T, Val(M), Val(Npoly)) do x
            Bessels.besseli0(β * sqrt(1 - x^2))
        end
        KaiserBesselKernelData{M}(Δx, σ, w, β, β², cs, gk)
    end
end

function Adapt.adapt_structure(to, g::KaiserBesselKernelData{M}) where {M}
    KaiserBesselKernelData{M}(
        g.Δx, g.σ, g.w, g.β, g.β²,
        adapt(to, g.cs),
        adapt(to, g.gk),
    )
end

KaiserBesselKernelData(::HalfSupport{M}, args...) where {M} =
    KaiserBesselKernelData{M}(args...)

function Base.show(io::IO, g::KaiserBesselKernelData{M}) where {M}
    (; β,) = g
    print(io, "KaiserBesselKernel(β = $β) with half-support M = $M")
end

function optimal_kernel(kernel::KaiserBesselKernel, h::HalfSupport{M}, Δx, σ; backend) where {M}
    T = typeof(Δx)
    β = if kernel.β === nothing
        # Set the optimal kernel shape parameter given the wanted support M and the oversampling
        # factor σ. See Potts & Steidl 2003, eq. (5.12).
        γ = 0.980  # empirical "safety factor" which slightly improves accuracy, as in FINUFFT (where γ = 0.976)
        T(M * π * (2 - 1 / σ) * γ)
    else
        T(kernel.β)
    end
    KaiserBesselKernelData(h, backend, Δx, β)
end

function evaluate_fourier_func(g::KaiserBesselKernelData)
    (; β², w,) = g
    function (k)
        q = w * k
        s = sqrt(β² - q^2)  # this is always real (assuming β ≥ Mπ)
        2 * w * sinh(s) / s
    end
end

function evaluate_kernel_func(g::KaiserBesselKernelData{M, T}) where {M, T}
    (; Δx, cs,) = g
    function (x)
        i, r = point_to_cell(x, Δx)
        X = r - T(i - 1)  # = (x - x[i]) / Δx = x / Δx - (i - 1)
        # @assert 0 ≤ X < 1
        values = evaluate_piecewise(X, cs)
        (; i, values,)
    end
end

# Not sure Bessels.besseli0 is optimised/overridden for GPUs. It works but I'm not sure it's
# optimal, since it uses the implementation in Bessels.jl which is not made for
# GPUs. An alternative would be to use functions from SpecialFunctions.jl, for which CUDA.jl
# redirects to CUDA functions. However, CUDA.jl currently doesn't wrap the cyl_bessel_i0
# function needed here (and also, SpecialFunctions doesn't provide a besseli0 function, but
# only a besseli function for arbitrary order).
# So, on CUDA, we use a package extension (ext/NonuniformFFTsCUDAExt.jl) to override this
# function, and call the CUDA-defined version instead.
_besseli0(x) = Bessels.besseli0(x)

@inline function _evaluate_kernel_direct(
        g::KaiserBesselKernelData{M, T}, i::Integer, r::T,
    ) where {M, T}
    (; β,) = g
    X = r - T(i - 1)  # = (x - x[i]) / Δx = x / Δx - (i - 1)
    # @assert 0 ≤ X < 1
    js = SVector(ntuple(identity, Val(2M)))
    ys = @. T(M - js + X) / M
    zs = @. 1 - ys^2
    s = @fastmath sqrt.(zs)  # the @fastmath avoids checking that z ≥ 0, returns NaN otherwise
    vals = _besseli0.(β * s)
    Tuple(vals)
end
