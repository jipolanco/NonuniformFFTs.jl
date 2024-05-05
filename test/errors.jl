using NonuniformFFTs
using Test

@testset "Error checks" begin
    @testset "ArgumentError: data size is too small" begin
        σ = 1.25
        m = HalfSupport(8)
        N = 12  # floor(σN) = 15 < 2M
        @test_throws ArgumentError PlanNUFFT(Float64, N; m, σ)
    end
end
