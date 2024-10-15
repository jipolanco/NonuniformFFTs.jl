export BackwardsKaiserBesselKernel

using Bessels: besseli0

backwards_kb_equivalent_variance(β) = sinh(β) / (β * cosh(β) - sinh(β))

@doc raw"""
    BackwardsKaiserBesselKernel <: AbstractKernel
    BackwardsKaiserBesselKernel([β])

Represents a backwards [Kaiser–Bessel](https://en.wikipedia.org/wiki/Kaiser_window#Definition) (KB)
spreading kernel.

This kernel basically results from swapping the [`KaiserBesselKernel`](@ref) spreading
function and its Fourier transform.
It has very similar properties to the Kaiser–Bessel kernel.

# Definition

```math
ϕ(x) = \frac{\sinh \left(β \sqrt{1 - x²} \right)}{π \sqrt{1 - x²}}
\quad \text{ for } |x| ≤ 1
```

where ``β`` is a shape factor.

# Fourier transform

```math
ϕ̂(k) = I₀ \left( \sqrt{β² - k²} \right)
```

where ``I₀`` is the zeroth-order [modified Bessel
function](https://en.wikipedia.org/wiki/Modified_Bessel_function#Modified_Bessel_functions:_I%CE%B1,_K%CE%B1)
of the first kind.

# Parameter selection

By default, the shape parameter is chosen to be [1]

```math
β = γ M π \left( 2 - \frac{1}{σ} \right)
```

where ``M`` is the kernel half-width and ``σ`` the oversampling factor.
Moreover, ``γ = 0.995`` is an empirical "safety factor", similarly to the one used by
FINUFFT [2], which slightly improves accuracy.

This default value can be overriden by explicitly passing a ``β`` value.

""" *
"""
# Implementation details

Since the evaluation of the hyperbolic sine functions can be costly, this kernel is
efficiently evaluated via an accurate piecewise polynomial approximation.
We use the same method originally proposed for FINUFFT [2] and later discussed by
Shamshirgar et al. [3].

[1] Potts & Steidl, SIAM J. Sci. Comput. **24**, 2013 (2003) \\
[2] Barnett, Magland & af Klinteberg, SIAM J. Sci. Comput. **41**, C479 (2019) \\
[3] Shamshirgar, Bagge & Tornberg, J. Chem. Phys. **154**, 164109 (2021)
"""
struct BackwardsKaiserBesselKernel{Beta <: Union{Nothing, Real}} <: AbstractKernel
    β :: Beta
end

BackwardsKaiserBesselKernel() = BackwardsKaiserBesselKernel(nothing)

struct BackwardsKaiserBesselKernelData{
        M, T <: AbstractFloat, ApproxCoefs <: NTuple,
        FourierCoefs <: AbstractVector{T},
    } <: AbstractKernelData{BackwardsKaiserBesselKernel, M, T}
    Δx :: T  # grid spacing
    σ  :: T  # equivalent kernel width (for comparison with Gaussian)
    w  :: T  # actual kernel half-width (= M * Δx)
    β  :: T  # KB parameter
    cs :: ApproxCoefs  # coefficients of polynomial approximation
    gk :: FourierCoefs

    function BackwardsKaiserBesselKernelData{M}(Δx::T, σ::T, w::T, β::T, cs, gk) where {M, T <: AbstractFloat}
        new{M, T, typeof(cs), typeof(gk)}(Δx, σ, w, β, cs, gk)
    end

    function BackwardsKaiserBesselKernelData{M}(backend::KA.Backend, Δx::T, β::T) where {M, T <: AbstractFloat}
        w = M * Δx
        σ = sqrt(backwards_kb_equivalent_variance(β)) * w
        gk = KA.allocate(backend, T, 0)
        Npoly = M + 4  # degree of polynomial is d = Npoly - 1
        cs = solve_piecewise_polynomial_coefficients(T, Val(M), Val(Npoly)) do x
            s = sqrt(1 - x^2)
            sinh(β * s) / (s * oftype(x, π))
        end
        BackwardsKaiserBesselKernelData{M}(Δx, σ, w, β, cs, gk)
    end
end

function Adapt.adapt_structure(to, g::BackwardsKaiserBesselKernelData{M}) where {M}
    BackwardsKaiserBesselKernelData{M}(
        g.Δx, g.σ, g.w, g.β,
        adapt(to, g.cs),
        adapt(to, g.gk),
    )
end

BackwardsKaiserBesselKernelData(::HalfSupport{M}, args...) where {M} =
    BackwardsKaiserBesselKernelData{M}(args...)

function Base.show(io::IO, g::BackwardsKaiserBesselKernelData{M}) where {M}
    (; β,) = g
    print(io, "BackwardsKaiserBesselKernel(β = $β) with half-support M = $M")
end

function optimal_kernel(kernel::BackwardsKaiserBesselKernel, h::HalfSupport{M}, Δx, σ; backend) where {M}
    T = typeof(Δx)
    β = if kernel.β === nothing
        # Set the optimal kernel shape parameter given the wanted support M and the oversampling
        # factor σ. See Potts & Steidl 2003, eq. (5.12).
        γ = 0.995  # empirical "safety factor" which slightly improves accuracy, as in FINUFFT (where γ = 0.976)
        T(M * π * (2 - 1 / σ) * γ)
    else
        T(kernel.β)
    end
    BackwardsKaiserBesselKernelData(h, backend, Δx, β)
end

function evaluate_fourier_func(g::BackwardsKaiserBesselKernelData)
    (; β, w,) = g
    function (k)
        q = w * k
        s = sqrt(β^2 - q^2)  # this is always real (assuming β ≥ Mπ)
        w * besseli0(s)
    end
end

function evaluate_kernel_func(g::BackwardsKaiserBesselKernelData{M, T}) where {M, T}
    (; w, Δx, cs,) = g
    function (x)
        i = point_to_cell(x, Δx)
        X = x / w - T(i - 1) / M  # source position relative to xs[i]
        # @assert 0 ≤ X < 1 / M
        values = evaluate_piecewise(X, cs)
        (; i, values,)
    end
end
