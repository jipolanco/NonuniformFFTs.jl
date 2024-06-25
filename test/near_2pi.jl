using Test
using NonuniformFFTs
using NonuniformFFTs: Kernels
using FFTW: fftfreq, rfftfreq

function type1_exact!(us, ks, xp, vp)
    fill!(us, 0)
    for i ∈ eachindex(xp, vp)
        for j ∈ eachindex(us)
            us[j] += vp[i] * cis(-xp[i] * ks[j])
        end
    end
    us
end

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

        us_exact = similar(us)
        ks = fftfreq(N, N)
        type1_exact!(us_exact, ks, xp, vp)

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

# Test similar issue when x = π - ε
@testset "Point near π" begin
    N = 16  # number of Fourier modes
    T = Float64
    x = T(π)
    x = prevfloat(x)
    xp = [x]
    vp = [3.4]

    @testset "Kernels.point_to_cell" begin
        Δx = T(2π) / 24
        i = Kernels.point_to_cell(x, Δx)
        @test (i - 1) * Δx ≤ x < i * Δx
    end

    plan_nufft = PlanNUFFT(T, N; m = HalfSupport(4), σ = 1.5)
    set_points!(plan_nufft, xp)
    us = Array{Complex{T}}(undef, size(plan_nufft))
    exec_type1!(us, plan_nufft, vp)

    us_exact = similar(us)
    ks = rfftfreq(N, N)
    type1_exact!(us_exact, ks, xp, vp)

    @test isapprox(us, us_exact; rtol = 1e-5)
end
