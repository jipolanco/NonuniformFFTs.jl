using Test
using Random: Random
using AbstractFFTs: fftfreq, rfftfreq
using JET: JET
using NonuniformFFTs

function test_plan_inference(args...; kws...)
    PlanNUFFT(args...; kws...)
end

# TODO support T <: Complex
function test_nufft_type1_1d(
        ::Type{T};
        kernel::Type{KernelType} = KaiserBesselKernel,
        N = 256,
        Np = 2 * N,
        m = HalfSupport(8),
        σ = 1.25,
    ) where {T <: AbstractFloat, KernelType}
    ks = rfftfreq(N, N)  # wavenumbers (= [0, 1, 2, ..., N÷2])

    # Generate some non-uniform random data
    rng = Random.Xoshiro(42)
    xp = rand(rng, real(T), Np) .* 2π  # non-uniform points in [0, 2π]
    vp = randn(rng, T, Np)             # random values at points

    # Compute "exact" non-uniform transform
    ûs_exact = zeros(Complex{T}, length(ks))
    for (i, k) ∈ pairs(ks)
        ûs_exact[i] = sum(zip(xp, vp)) do (x, v)
            v * cis(-k * x)
        end
    end

    # Compute NUFFT
    ûs = Array{Complex{T}}(undef, length(ks))
    plan_nufft = PlanNUFFT(T, N, m; σ, kernel = KernelType)
    NonuniformFFTs.set_points!(plan_nufft, xp)
    NonuniformFFTs.exec_type1!(ûs, plan_nufft, vp)

    # Check results
    err = sqrt(sum(splat((a, b) -> abs2(a - b)), zip(ûs, ûs_exact)) / sum(abs2, ûs_exact))

    # Inference tests
    JET.@test_opt NonuniformFFTs.set_points!(plan_nufft, xp)
    JET.@test_opt NonuniformFFTs.exec_type1!(ûs, plan_nufft, vp)

    M = NonuniformFFTs.Kernels.half_support(m)

    if σ ≈ 1.25 && T === Float64
        err_min_kb = 4e-12  # error reaches a minimum at ~2e-12 for M = 10
        if KernelType === KaiserBesselKernel
            # This seems to work for KaiserBesselKernel and 4 ≤ M ≤ 10
            @test err < max(10.0^(-1.16 * M) * 1.05, err_min_kb)
        elseif KernelType === BackwardsKaiserBesselKernel
            # This one seems to have slightly smaller error!
            # Verified for 4 ≤ M ≤ 10.
            @test err < max(10.0^(-1.20 * M), err_min_kb)
        end
    elseif σ ≈ 2 && T === Float64
        err_max_kb = max(5 * 10.0^(-1.9 * M), 3e-14)  # error "plateaus" at ~2e-14 for M ≥ 8
        if KernelType === KaiserBesselKernel
            @test err < err_max_kb
        elseif KernelType === BackwardsKaiserBesselKernel
            @test err < err_max_kb
        elseif KernelType === GaussianKernel
            @test err < 10.0^(-0.95 * M) * 0.8
        elseif KernelType === BSplineKernel
            @test err < 10.0^(-0.98 * M) * 0.4
        end
    end

    nothing
end

@testset "Type 1 NUFFTs" begin
    for M ∈ 4:10
        m = HalfSupport(M)
        σ = 1.25
        @testset "$kernel (m = $M, σ = $σ)" for kernel ∈ (KaiserBesselKernel, BackwardsKaiserBesselKernel)
            test_nufft_type1_1d(Float64; m, σ, kernel)
        end
        σ = 2.0
        @testset "$kernel (m = $M, σ = $σ)" for kernel ∈ (KaiserBesselKernel, BackwardsKaiserBesselKernel, GaussianKernel, BSplineKernel)
            test_nufft_type1_1d(Float64; m, σ, kernel)
        end
    end
end
