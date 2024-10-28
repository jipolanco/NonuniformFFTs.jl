# Test the AbstractNFFTs.jl interface and compare results with NFFT.jl.

using Test
using Random: Random
using NonuniformFFTs
using NFFT: NFFT, size_in, size_out

function compare_with_nfft(Ns; Np = 1000)
    rng = Random.Xoshiro(43)
    T = Float64                           # must be a real data type (Float32, Float64)
    d = length(Ns)                        # number of dimensions
    xp = rand(rng, T, (d, Np)) .- T(0.5)  # non-uniform points in [-1/2, 1/2)ᵈ; must be given as a (d, Np) matrix
    vp = randn(rng, Complex{T}, Np)       # random values at points (must be complex)

    reltol = 1e-9
    window = :kaiser_bessel

    p = NonuniformFFTs.NFFTPlan(xp, Ns; reltol, window,)
    p_nfft = NFFT.NFFTPlan(xp, Ns; reltol, window,)

    @test startswith(repr(p), "NonuniformFFTs.NFFTPlan{$T, $d} wrapping a PlanNUFFT:")  # test pretty-printing

    @test size_in(p) === size_in(p_nfft)
    @test size_out(p) === size_out(p_nfft)

    # Test type-1 (adjoint) transform
    us = adjoint(p) * vp
    us_nfft = adjoint(p_nfft) * vp
    @test us ≈ us_nfft

    # Test type-2 (forward) transform
    wp = p * us
    wp_nfft = p_nfft * us
    @test wp ≈ wp_nfft

    nothing
end

@testset "Comparison with NFFT.jl" begin
    dims_tested = [
        (512,),     # 1D
        (64, 81),   # 2D (we test odd sizes as well)
    ]
    @testset "Dimensions: $dims" for dims ∈ dims_tested
        compare_with_nfft(dims)
    end
end
