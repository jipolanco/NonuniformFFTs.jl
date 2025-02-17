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

function test_uniform_points(::Type{T}, N; σ = 1.25) where {T <: Number}
    Np = N
    rng = Random.Xoshiro(42)
    Tr = real(T)
    xp = collect(range(Tr(0), Tr(2π); length = N + 1)[1:N])
    vp = randn(rng, T, Np)

    if T <: Real
        ûs_fft = FFTW.rfft(vp)
        ûs_fft[end] = 0  # zero-out Nyquist mode to avoid comparison issues
        vp = FFTW.irfft(ûs_fft, N)
    elseif T <: Complex
        ûs_fft = FFTW.fft(vp)
    end

    ûs = similar(ûs_fft)

    plan_nufft = @inferred PlanNUFFT(T, N, HalfSupport(8); σ)
    set_points!(plan_nufft, xp)
    exec_type1!(ûs, plan_nufft, vp)

    err_type1 = l2_error(ûs, ûs_fft)
    @show err_type1
    @test err_type1 < 4e-10

    if T <: Real
        vp_expected = FFTW.brfft(ûs_fft, N)  # this is the unnormalised backwards FFT
    elseif T <: Complex
        vp_expected = FFTW.bfft(ûs_fft)  # this is the unnormalised backwards FFT
    end
    @assert vp_expected ≈ vp * N
    vp_bis = similar(vp)
    exec_type2!(vp_bis, plan_nufft, ûs)
    err_type2 = l2_error(vp_bis, vp_expected)
    @show err_type2
    @test err_type2 < 5e-10

    nothing
end

@testset "Uniform points" begin
    @testset "T = $T" for T ∈ (Float64, ComplexF64)
        test_uniform_points(T, 256)
    end
end
