using FFTW: FFTW

function chebyshev_create_fftw_plan(cs::AbstractVector; flags = FFTW.ESTIMATE)
    # Using kind = FFTW.REDFT10 corresponds to "the" discrete cosine transform (DCT),
    # and requires Chebyshev nodes in x[i] = [cos(π * (i - 1/2) / N) for i ∈ 1:N].
    FFTW.plan_r2r!(cs, FFTW.REDFT10; flags)
end

function chebyshev_approximate!(
        f::F, cs::AbstractVector,
        plan = chebyshev_create_fftw_plan(cs),
    ) where {F <: Function}
    Base.require_one_based_indexing(cs)
    T = eltype(cs)
    N = length(cs)

    # 1. Evaluate function on Chebyshev nodes (translated and rescaled from [-1, 1] to [0, 1]).
    for i ∈ 1:N
        x = (cospi(T(i - 1/2) / N) + 1) / 2  # in [0, 1]
        cs[i] = f(x)
    end

    # 2. Obtain Chebyshev coefficients via discrete cosine transform (DCT).
    # We assume this is an in-place plan.
    plan * cs

    # 3. Normalise DCT and rescale first coefficient.
    cs ./= N
    cs[begin] /= 2

    cs
end

# Similar to evaluation_points, but restricts the domain to [0, 1] instead of [-1, 1].
# We can do this since kernels are symmetric (and Chebyshev interpolations are performed
# on [0, 1] only).
function evaluation_points_chebyshev(::Val{M}, X) where {M}
    xs_right = ntuple(Val(M)) do j
        X + (M - j) / M  # in [0, 1]
    end
    xs_left = ntuple(Val(M)) do j
        # Note: we evaluate at -x instead of x (which is negative), taking into account that
        # the kernel is symmetric.
        j / M - X  # in [0, 1]
    end
    (xs_right..., xs_left...)
end

# Evaluate truncated Chebyshev series using the usual (forward) recurrence definition of
# Chebyshev polynomials.
# Note that `x` can be either a scalar or a vector, to evaluate at several locations at
# once.
# In the second case, using an SVector (or simply a tuple) can lead to huge performance
# gains due to vectorisation.
function chebyshev_evaluate(cs::AbstractVector, x_in)  # x_in is in [0, 1]
    x = @. 2 * x_in - 1  # in [-1, 1]
    Tₙ₋₁ = @. false * x + true  # = 1 (but works for scalars, arrays and tuples)
    Tₙ = x
    @inbounds y = @. cs[begin] + cs[begin + 1] * x
    @fastmath for i ∈ eachindex(cs)[3:end]
        @inbounds c = cs[i]
        T = @. 2 * x * Tₙ - Tₙ₋₁
        y = @. y + c * T
        Tₙ₋₁, Tₙ = Tₙ, T
    end
    y
end

# Similar to above but using the Clenshaw algorithm (backwards recurrence).
# It seems to give the same results and have roughly the same performance.
function chebyshev_evaluate_clenshaw(cs::AbstractVector, x_in)
    x = @. 2 * x_in - 1
    bₙ = @. false * x  # = 0 (but works for scalars, arrays and tuples)
    bₙ₊₁ = @. false * x
    @inbounds c = cs[end]
    @inbounds inds = reverse(eachindex(cs)[begin:end-1])
    @fastmath for i ∈ inds
        b = @. c + 2 * x * bₙ - bₙ₊₁
        bₙ, bₙ₊₁ = b, bₙ
        @inbounds c = cs[i]
    end
    b₁, b₂ = bₙ, bₙ₊₁
    @. c - b₂ + b₁ * x
end
