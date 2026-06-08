using Test
using NonuniformFFTs
using NonuniformFFTs: Kernels
using FFTW: fftfreq, rfftfreq
using TimerOutputs: TimerOutput

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
    @testset "Determination of cell index ($T)" for T in (Float32, Float64)
        L = 2 * T(π)
        Linv = 1 / L
        Ns = 400:100:10000
        inds_dx = similar(Ns)
        inds_dx_inv = similar(Ns)
        inds_N_L = similar(Ns)
        inds_L_N = similar(Ns)
        inds_Linv_N = similar(Ns)

        for (n, N) in pairs(Ns)
            Δx = L / N
            Δx_inv = N / L
            x = prevfloat(L)
            inds_dx[n] = unsafe_trunc(Int, x / Δx) + 1
            inds_dx_inv[n] = unsafe_trunc(Int, x * Δx_inv) + 1
            inds_N_L[n] = unsafe_trunc(Int, (x * N) / L) + 1
            # The following two ensure that the resulting index is inbounds (in 1:N)
            inds_L_N[n] = unsafe_trunc(Int, (x / L) * N) + 1
            inds_Linv_N[n] = unsafe_trunc(Int, (x * Linv) * N) + 1
        end

        @test count(inds_dx .!= Ns) > 0
        @test count(inds_dx_inv .!= Ns) > 0
        @test count(inds_N_L .!= Ns) > 0
        @test count(inds_L_N .!= Ns) == 0  # this one never fails in the tested range
        @test count(inds_Linv_N .!= Ns) == 0  # this one never fails in the tested range
    end

    @testset "NUFFT" begin
        L = 2π

        x = prevfloat(L)  # such that x < L
        xp = [x]
        vp = [4.2 + 3im]
        T = eltype(vp)

        # These parameters allow to reproduce old issue when a point is very close to 2π.
        N = 32
        plan = PlanNUFFT(T, N; m = HalfSupport(8), σ = 1.5, block_size = 16)
        @test @inferred((p -> p.timer)(plan)) isa TimerOutput
        set_points!(plan, xp)

        us = Array{T}(undef, N)
        exec_type1!(us, plan, vp)

        us_exact = similar(us)
        ks = fftfreq(N, N)
        type1_exact!(us_exact, ks, xp, vp)

        @test isapprox(us, us_exact; rtol = 1e-11)
    end

    @testset "Kernels.point_to_cell" begin
        N = 3
        L = Kernels.domain_period(Float64)  # = 2π
        Δx = L / N
        let x = prevfloat(L / 3)
            @test Kernels.point_to_cell(x, N)[1] == 1
        end
        let x = prevfloat(2 * L / 3)
            @test Kernels.point_to_cell(x, N)[1] == 2
        end
        let x = prevfloat(L)
            @test Kernels.point_to_cell(x, N)[1] == 3
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
        N = 24
        Δx = Kernels.domain_period(T) / N
        i, r = Kernels.point_to_cell(x, N)
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
