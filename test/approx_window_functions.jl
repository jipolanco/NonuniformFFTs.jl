# Test direct ("exact") and approximate evaluation of window functions.
# This is particularly useful for testing approximations based on piecewise polynomials.

using NonuniformFFTs
using NonuniformFFTs.Kernels
using StaticArrays
using Test

function test_kernel(kernel; σ = 1.5, m = HalfSupport(4), N = 256)
    backend = CPU()
    Δx = 2π / N
    g = Kernels.optimal_kernel(kernel, m, Δx, σ; backend)
    xs = range(0.8, 2.2; length = 1000) .* Δx

    for x ∈ xs
        a = Kernels.evaluate_kernel(g, x)
        b = Kernels.evaluate_kernel_direct(g, x)
        @test a.i == b.i  # same bin
        @test SVector(a.values) ≈ SVector(b.values) rtol=1e-7  # TODO: rtol should depend on (M, σ, kernel)
    end

    nothing
end

@testset "Kernel approximations" begin
    kernels = (
        BSplineKernel(),
        GaussianKernel(),
        KaiserBesselKernel(),
        BackwardsKaiserBesselKernel(),
    )
    @testset "$(nameof(typeof(kernel)))" for kernel ∈ kernels
        test_kernel(kernel)
    end
end
