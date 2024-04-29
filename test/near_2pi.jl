using Test
using NonuniformFFTs
using NonuniformFFTs: Kernels
using FFTW: fftfreq

# Test that roundoff errors are properly accounted for when a point is very close to 2π.

@testset "Point near 2π" begin
    L = 2π

    @testset "NUFFT" begin
        x = prevfloat(L)  # such that x < L
        xp = [x]
        vp = [4.2 + 3im]
        T = eltype(vp)

        # NOTE: these parameters allow to reproduce issue when a point is very close to 2π.
        N = 32
        plan = PlanNUFFT(T, N; m = HalfSupport(8), σ = 1.5, block_size = 16)
        set_points!(plan, xp)

        us = Array{T}(undef, N)
        exec_type1!(us, plan, vp)

        us_exact = fill!(similar(us), 0)
        ks = fftfreq(N, N)
        for i ∈ eachindex(xp, vp)
            for j ∈ eachindex(us_exact)
                us_exact[j] += vp[i] * cis(-xp[i] * ks[j])
            end
        end

        @test isapprox(us, us_exact; rtol = 1e-11)
    end

    @testset "Kernels.point_to_cell" begin
        Δx = 2π / 3
        let x = prevfloat(L / 3)
            @test Kernels.point_to_cell(x, Δx) == 1
        end
        let x = prevfloat(2 * L / 3)
            @test Kernels.point_to_cell(x, Δx) == 2
        end
        let x = prevfloat(L)
            @test Kernels.point_to_cell(x, Δx) == 3
        end
    end
end
