using StaticArrays: MVector
using Base.Cartesian: @ntuple

export GaussianKernel

@doc raw"""
    GaussianKernel <: AbstractKernel
    GaussianKernel([ℓ/Δx])

Represents a truncated Gaussian spreading kernel.

# Definition

```math
ϕ(x) = e^{-x² / 2ℓ²}
```

where ``ℓ`` is the characteristic width of the kernel.

# Fourier transform

```math
ϕ̂(k) = \sqrt{2πσ²} e^{-ℓ² k² / 2}
```

# Parameter selection

By default, given a kernel half-width ``M``, an oversampling factor ``σ`` and the oversampling grid
spacing ``Δx``, the characteristic width ``ℓ`` is chosen as [1]

```math
ℓ² = Δx² \frac{σ}{2σ - 1} \frac{M}{π}
```

This default value can be overriden by explicitly passing a value of the wanted normalised width ``ℓ/Δx``.

""" *
"""
# Implementation details

In the implementation, this kernel is efficiently evaluated using the fast Gaussian gridding
method proposed by Greengard & Lee [2].

[1] Potts & Steidl, SIAM J. Sci. Comput. **24**, 2013 (2003) \\
[2] Greengard & Lee, SIAM Rev. **46**, 443 (2004)
"""
struct GaussianKernel{Width <: Union{Nothing, Real}} <: AbstractKernel
    ℓ :: Width
end

GaussianKernel() = GaussianKernel(nothing)

"""
    GaussianKernelData(HalfSupport(M), Δx, α)

Constructs a Gaussian kernel with standard deviation `σ = α Δx` and half-support `M`, for a
grid of step `Δx`.
"""
struct GaussianKernelData{M, T <: AbstractFloat} <: AbstractKernelData{GaussianKernel, M, T}
    Δx :: T
    σ  :: T
    τ  :: T
    cs :: NTuple{M, T}  # precomputed exponentials
    gk :: Vector{T}     # values in uniform Fourier grid
    function GaussianKernelData{M}(Δx::T, α::T) where {M, T <: AbstractFloat}
        σ = α * Δx
        τ = 2 * σ^2
        cs = ntuple(Val(M)) do i
            x = i * Δx
            exp(-x^2 / τ)
        end
        gk = Vector{T}(undef, 0)
        new{M, T}(Δx, σ, τ, cs, gk)
    end
end

GaussianKernelData(::HalfSupport{M}, args...) where {M} = GaussianKernelData{M}(args...)

function optimal_kernel(kernel::GaussianKernel, h::HalfSupport{M}, Δx, σ) where {M}
    T = typeof(Δx)
    ℓ = if kernel.ℓ === nothing
        # Set the optimal kernel shape parameter given the wanted support M and the oversampling
        # factor σ. See Potts & Steidl 2003, eq. (5.9).
        T(sqrt(σ * M / (2 * σ - 1) / π))
    else
        T(kernel.ℓ)
    end
    GaussianKernelData(h, Δx, ℓ)
end

function evaluate_fourier(g::GaussianKernelData, k::Number)
    (; τ,) = g
    exp(-τ * k^2 / 4) * sqrt(π * τ)  # = exp(-σ² k² / 2) * sqrt(2πσ²)
end

# Fast Gaussian gridding following Greengard & Lee, SIAM Rev. 2004.
@inline @fastmath function evaluate_kernel(g::GaussianKernelData{M}, x, i::Integer) where {M}
    # Evaluate in-between grid points xs[(i - M):(i + M)].
    # Note: xs[j] = (j - 1) * Δx
    (; τ, Δx, cs,) = g
    X = x - (i - 1) * Δx  # source position relative to xs[i]
    # @assert 0 ≤ X < i * Δx
    a = exp(-X^2 / τ)
    b = exp(2X * Δx / τ)
    values = gaussian_gridding(a, b, cs)
    (; i, values,)
end

@inline function gaussian_gridding(a, b, cs::NTuple{M}) where {M}
    if @generated
        v₀ = Symbol(:v_, M)
        ex = quote
            $v₀ = a
            bpow = one(b)
        end
        @inbounds for m = 1:(M - 1)
            v₋ = Symbol(:v_, M - m)
            v₊ = Symbol(:v_, M + m)
            ex = quote
                $ex
                bpow *= b
                $v₋ = a * cs[$m] / bpow
                $v₊ = a * cs[$m] * bpow
            end
        end
        L = 2M
        v_end = Symbol(:v_, L)
        quote
            $ex
            $v_end = a * cs[$M] * bpow * b
            @ntuple $L v
        end
    else
        L = 2M
        bpow = one(b)
        vs = MVector{L, typeof(a)}(undef)
        vs[M] = a
        for m = 1:(M - 1)
            bpow *= b
            vs[M - m] = a * cs[m] / bpow
            vs[M + m] = a * cs[m] * bpow
        end
        vs[2M] = a * cs[M] * bpow * b
        Tuple(vs)
    end
end
