export BackwardsKaiserBesselKernel

using Bessels: besseli0

backwards_kb_equivalent_variance(β) = sinh(β) / (β * cosh(β) - sinh(β))

struct BackwardsKaiserBesselKernel{M, T <: AbstractFloat} <: AbstractKernel{M}
    Δx :: T  # grid spacing
    σ  :: T  # equivalent kernel width (for comparison with Gaussian)
    w  :: T  # actual kernel half-width (= M * Δx)
    β  :: T  # KB parameter
    sinh_β :: T
    gk :: Vector{T}
    function BackwardsKaiserBesselKernel{M}(Δx::T, β::T) where {M, T <: AbstractFloat}
        w = M * Δx
        σ = sqrt(backwards_kb_equivalent_variance(β)) * w
        sinh_β = sinh(β)
        gk = Vector{T}(undef, 0)
        new{M, T}(Δx, σ, w, β, sinh_β, gk)
    end
end

BackwardsKaiserBesselKernel(::HalfSupport{M}, args...) where {M} =
    BackwardsKaiserBesselKernel{M}(args...)

function evaluate_fourier(g::BackwardsKaiserBesselKernel, k::Number)
    (; β, w,) = g
    q = w * k
    # s = sqrt(complex(β^2 - q^2))  # TODO is this always real?
    s = sqrt(β^2 - q^2)  # we assume this is always real (requires large enough β)
    w * π * besseli0(s) / sinh(β)
end

function evaluate_kernel(g::BackwardsKaiserBesselKernel{M}, x, i::Integer) where {M}
    # Evaluate in-between grid points xs[(i - M):(i + M)].
    # Note: xs[j] = (j - 1) * Δx
    (; β, w, sinh_β,) = g
    X = x / w - (i - 1) / M  # source position relative to xs[i]
    # @assert 0 ≤ X < 1 / M
    # We take advantage of (automatic) SIMD vectorisation by computing sqrt and exp at once
    # over all points.
    # Note that we explicitly compute sinh(z) = (exp(z) + exp(-z)) / 2 instead of Julia's
    # sinh, which might be more accurate for some z's but creates branches and is thus
    # slower.
    xs = evaluation_points(Val(M), X)
    sq = @. sqrt(1 - xs^2)
    es = @. exp(β * sq)
    values = @. (es + 1/es) / (2 * sinh_β * sq)
    (; i, values,)
end
