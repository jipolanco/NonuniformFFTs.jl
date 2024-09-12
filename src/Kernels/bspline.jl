export
    BSplineKernel

using Base.Cartesian: @ntuple, @nexprs

@doc raw"""
    BSplineKernel <: AbstractKernel
    BSplineKernel()

Represents a B-spline spreading kernel.

The order ``n`` of the B-spline is directly related to the kernel half-width ``M`` by
``n = 2M`` (the polynomial degree is ``n - 1``).

# Definition

A B-spline of order ``n`` may be defined via its Fourier transform:

```math
ϕ̂(k) = Δx \operatorname{sinc}^n \left( \frac{k Δx}{2} \right)
```

where ``Δx`` is the spacing of the oversampled grid and ``\operatorname{sinc}``
is the [unnormalised sinc function](https://en.wikipedia.org/wiki/Sinc_function).

# Implementation details

In the implementation, this kernel is evaluated using
[de Boor's algorithm](https://en.wikipedia.org/wiki/De_Boor%27s_algorithm).
"""
struct BSplineKernel <: AbstractKernel end

"""
    BSplineKernelData(HalfSupport(M), Δx)

Constructs a B-spline kernel with half-support `M` for a grid of step `Δx`.

The B-spline order is simply `n = 2M`.
Note that the polynomial degree is `n - 1`.
In other words, setting `M = 2` corresponds to cubic B-splines.

For performance reasons, the separation ``Δt`` between B-spline knots is set to
be equal to the grid step ``Δx``.
This means that the resulting variance of the B-spline kernels is fixed to
``σ^2 = (n / 12) Δt^2 = (M / 6) Δx^2``.
"""
struct BSplineKernelData{
        M, T <: AbstractFloat,
        FourierCoefs <: AbstractVector{T},
    } <: AbstractKernelData{BSplineKernel, M, T}
    σ  :: T
    Δt :: T          # knot separation
    gk :: FourierCoefs  # values in uniform Fourier grid
    function BSplineKernelData{M}(backend::KA.Backend, Δx::Real) where {M}
        Δt = Δx
        σ = sqrt(M / 6) * Δt
        T = eltype(Δt)
        gk = KA.allocate(backend, T, 0)
        new{M, T, typeof(gk)}(T(σ), Δt, gk)
    end
end

gridstep(g::BSplineKernelData) = g.Δt  # assume Δx = Δt

BSplineKernelData(::HalfSupport{M}, args...) where {M} = BSplineKernelData{M}(args...)

function Base.show(io::IO, ::BSplineKernelData{M}) where {M}
    print(io, "BSplineKernel() with half-support M = $M")
end

# Here we ignore the oversampling factor, this kernel is not very adjustable...
optimal_kernel(::BSplineKernel, h::HalfSupport, Δx, σ; backend) =
    BSplineKernelData(h, backend, Δx)

"""
    order(::BSplineKernelData{M})

Returns the order `n = 2M` of the B-spline kernel.

Note: the polynomial degree is `n - 1`.
"""
order(::BSplineKernelData{M}) where {M} = 2M

function evaluate_kernel(g::BSplineKernelData{M}, x, i::Integer) where {M}
    # The integral of a single B-spline, using its standard definition, is Δt.
    # This can be shown using the partition of unity property.
    (; Δt,) = g
    x′ = i - (x / Δt)  # normalised coordinate, 0 < x′ ≤ 1 (this assumes Δx = Δt)
    # @assert 0 ≤ x′ ≤ 1
    k = 2M  # B-spline order
    values = bsplines_evaluate_all(x′, Val(k), typeof(Δt))
    (; i, values,)
end

# This should work on CPU and GPU.
function evaluate_fourier!(g::BSplineKernelData{M}, gk::AbstractVector, ks::AbstractVector) where {M}
    (; Δt,) = g
    @assert eachindex(gk) == eachindex(ks)
    map!(gk, ks) do k
        kh = k * Δt / 2
        s = sin(kh) / kh
        n = 2M
        ifelse(iszero(k), one(s), s^n) * Δt
    end
end

# Adapted from BSplineKit.jl.
#
# Simplifications and modifications:
# - assume uniform knot interval Δt = 1
# - assume infinite knot vector -> t[i] = i
# - assume 0 ≤ x ≤ 1
#
# This returns a length-k tuple with the B-splines
# {b[0], b[-1], b[-2], …, b[-k + 1]} (in decreasing order), all evaluated at `x`.
#
# Equivalently, since all B-splines are a translation of each other, this function returns
# b[-M](x + j) for j ∈ [-M, M - 1], where M = k/2 (this time in increasing order).
function bsplines_evaluate_all(
        x::Real, ::Val{k}, ::Type{T},
    ) where {k, T}
    if @generated
        @assert k ≥ 1
        ex = quote
            bs_1 = (one(T),)
        end
        for q ∈ 2:k
            bp = Symbol(:bs_, q - 1)
            bq = Symbol(:bs_, q)
            α = one(T) / (q - 1)
            ex = quote
                $ex
                x′ = x
                Δs = @ntuple $(q - 1) j -> begin
                    val = $T($α * x′)
                    x′ += 1
                    val
                end
                $bq = bsplines_evaluate_step(Δs, $bp, Val($q))
            end
        end
        bk = Symbol(:bs_, k)
        quote
            $ex
            return $bk
        end
    else
        bsplines_evaluate_all_alt(x, Val(k), T)
    end
end

@inline @generated function bsplines_evaluate_step(Δs, bp, ::Val{k}) where {k}
    ex = quote
        @inbounds b_1 = Δs[1] * bp[1]
    end
    for j = 2:(k - 1)
        bj = Symbol(:b_, j)
        ex = quote
            $ex
            @inbounds $bj = (1 - Δs[$j - 1]) * bp[$j - 1] + Δs[$j] * bp[$j]
        end
    end
    b_last = Symbol(:b_, k)
    quote
        $ex
        @inbounds $b_last = (1 - Δs[$k - 1]) * bp[$k - 1]
        @ntuple $k b
    end
end

@inline function bsplines_evaluate_all_alt(
        x::Real, ::Val{k}, ::Type{T},
    ) where {k, T}
    # @assert 0 ≤ x < 1
    # @assert k ≥ 1
    bq = MVector{k, T}(undef)
    Δs = MVector{k - 1, T}(undef)
    bq[1] = one(T)
    @inbounds for q′ ∈ 1:(k - 1)
        q = q′ + 1
        α = one(T) / q
        x′ = x
        for j ∈ 1:q′
            Δs[j] = α * x′
            x′ += 1
        end
        bp = bq[1]
        Δp = Δs[1]
        bq[1] = Δp * bp
        for j = 2:q′
            bpp, bp = bp, bq[j]
            Δpp, Δp = Δp, Δs[j]
            bq[j] = Δp * bp + (1 - Δpp) * bpp
        end
        bq[q] = (1 - Δp) * bp
    end
    Tuple(bq)
end
