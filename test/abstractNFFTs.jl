# Test the AbstractNFFTs.jl interface and compare results with NFFT.jl.

using Test
using NonuniformFFTs
using NFFT: NFFT, size_in, size_out

Ns = (64, 81)  # number of Fourier modes in each direction (test odd sizes as well)
Np = 1000      # number of non-uniform points

T = Float64                      # must be a real data type (Float32, Float64)
d = length(Ns)                   # number of dimensions (d = 2 here)
xp = rand(T, (d, Np)) .- T(0.5)  # non-uniform points in [-1/2, 1/2)ᵈ; must be given as a (d, Np) matrix
vp = randn(Complex{T}, Np)       # random values at points (must be complex)

reltol = 1e-9
window = :kaiser_bessel

p = PlanNUFFT(xp, Ns)
p_nfft = NFFT.NFFTPlan(xp, Ns)

@testset "Comparison with NFFT.jl" begin
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
end
