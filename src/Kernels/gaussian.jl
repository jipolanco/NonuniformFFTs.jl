using StaticArrays: MVector
using Base.Cartesian: @ntuple

export GaussianKernel

"""
    GaussianKernel(HalfSupport(M), Δx, α)

Constructs a Gaussian kernel with standard deviation `σ = α Δx` and half-support `M`, for a
grid of step `Δx`.
"""
struct GaussianKernel{M, T <: AbstractFloat} <: AbstractKernel{M, T}
    Δx :: T
    σ  :: T
    τ  :: T
    cs :: NTuple{M, T}  # precomputed exponentials
    gk :: Vector{T}     # values in uniform Fourier grid
    function GaussianKernel{M}(Δx::T, α::T) where {M, T <: AbstractFloat}
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

GaussianKernel(::HalfSupport{M}, args...) where {M} = GaussianKernel{M}(args...)

function optimal_kernel(::Type{GaussianKernel}, h::HalfSupport{M}, Δx, σ) where {M}
    # Set the optimal kernel shape parameter given the wanted support M and the oversampling
    # factor σ. See Potts & Steidl 2003, eq. (5.9).
    α² = oftype(Δx, σ * M / (2 * σ - 1) / π)
    GaussianKernel(h, Δx, sqrt(α²))
end

function evaluate_fourier(g::GaussianKernel, k::Number)
    (; τ,) = g
    exp(-τ * k^2 / 4) * sqrt(π * τ)  # = exp(-σ² k² / 2) * sqrt(2πσ²)
end

# Fast Gaussian gridding following Greengard & Lee, SIAM Rev. 2004.
@inline @fastmath function evaluate_kernel(g::GaussianKernel{M}, x, i::Integer) where {M}
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
