# Check that a transform on uniform points is equal to the corresponding output from FFTW.
# Note that this equivalence is somewhat arbitrary and is a choice we make.

using Test
using Random: Random
using FFTW: FFTW
using NonuniformFFTs

function l2_error(us, vs)
    err = sum(zip(us, vs)) do (u, v)
        abs2(u - v)
    end
    norm = sum(abs2, vs)
    sqrt(err / norm)
end

function test_uniform_points(::Type{T}, N; σ = 1.25) where {T <: AbstractFloat}
    Np = N
    rng = Random.Xoshiro(42)
    xp = range(T(0), T(2π); length = N + 1)[1:N]
    vp = randn(rng, T, Np)

    ûs_fft = FFTW.rfft(vp)

    # Zero-out Nyquist mode to avoid comparison issues.
    ûs_fft[end] = 0
    vp = FFTW.irfft(ûs_fft, N)

    ûs = similar(ûs_fft)

    plan_nufft = @inferred PlanNUFFT(T, N, HalfSupport(8); σ)
    set_points!(plan_nufft, xp)
    exec_type1!(ûs, plan_nufft, vp)

    err_type1 = l2_error(ûs, ûs_fft)
    @show err_type1
    @test err_type1 < 3e-10

    vp_expected = FFTW.brfft(ûs_fft, N)  # this is the unnormalised backwards FFT
    @assert vp_expected ≈ vp * N
    vp_bis = similar(vp)
    exec_type2!(vp_bis, plan_nufft, ûs)
    err_type2 = l2_error(vp_bis, vp_expected)
    @show err_type2
    @test err_type2 < 3e-10

    nothing
end

@testset "Uniform points" begin
    test_uniform_points(Float64, 256)
end
