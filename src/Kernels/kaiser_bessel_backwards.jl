export BackwardsKaiserBesselKernel

using Bessels: besseli0

backwards_kb_equivalent_variance(β) = sinh(β) / (β * cosh(β) - sinh(β))

struct BackwardsKaiserBesselKernel <: AbstractKernel end

struct BackwardsKaiserBesselKernelData{
        M, T <: AbstractFloat, ApproxCoefs <: AbstractArray{T},
    } <: AbstractKernelData{BackwardsKaiserBesselKernel, M, T}
    Δx :: T  # grid spacing
    σ  :: T  # equivalent kernel width (for comparison with Gaussian)
    w  :: T  # actual kernel half-width (= M * Δx)
    β  :: T  # KB parameter
    sinh_β :: T
    cs :: ApproxCoefs  # coefficients of polynomial approximation
    gk :: Vector{T}
    function BackwardsKaiserBesselKernelData{M}(Δx::T, β::T) where {M, T <: AbstractFloat}
        w = M * Δx
        σ = sqrt(backwards_kb_equivalent_variance(β)) * w
        sinh_β = sinh(β)
        gk = Vector{T}(undef, 0)
        Npoly = M + 4  # degree of polynomial is d = Npoly - 1
        cs = solve_piecewise_polynomial_coefficients(T, Val(M), Val(Npoly)) do x
            s = sqrt(1 - x^2)
            sinh(β * s) / (s * sinh(β))
        end
        new{M, T, typeof(cs)}(Δx, σ, w, β, sinh_β, cs, gk)
    end
end

BackwardsKaiserBesselKernelData(::HalfSupport{M}, args...) where {M} =
    BackwardsKaiserBesselKernelData{M}(args...)

function optimal_kernel(::BackwardsKaiserBesselKernel, h::HalfSupport{M}, Δx, σ) where {M}
    # Set the optimal kernel shape parameter given the wanted support M and the oversampling
    # factor σ. See Potts & Steidl 2003, eq. (5.12).
    γ = 0.995  # empirical "safety factor" which slightly improves accuracy, as in FINUFFT (where γ = 0.976)
    β = oftype(Δx, M * π * (2 - 1 / σ)) * γ
    BackwardsKaiserBesselKernelData(h, Δx, β)
end

function evaluate_fourier(g::BackwardsKaiserBesselKernelData, k::Number)
    (; β, w,) = g
    q = w * k
    s = sqrt(β^2 - q^2)  # this is always real (assuming β ≥ Mπ)
    w * π * besseli0(s) / sinh(β)
end

function evaluate_kernel(g::BackwardsKaiserBesselKernelData{M}, x, i::Integer) where {M}
    # Evaluate in-between grid points xs[(i - M):(i + M)].
    # Note: xs[j] = (j - 1) * Δx
    (; β, w, sinh_β,) = g
    X = x / w - (i - 1) / M  # source position relative to xs[i]
    # @assert 0 ≤ X < 1 / M
    values = evaluate_piecewise(X, g.cs)
    (; i, values,)
end
