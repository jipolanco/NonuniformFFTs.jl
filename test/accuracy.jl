using Test
using Random: Random
using AbstractFFTs: fftfreq, rfftfreq
using JET: JET
using NonuniformFFTs

function check_nufft_error(::Type{Float64}, ::KaiserBesselKernel, ::HalfSupport{M}, σ, err) where {M}
    if σ ≈ 1.25
        err_min_kb = 4e-12  # error reaches a minimum at ~2e-12 for M = 10
        # This seems to work for KaiserBesselKernel and 4 ≤ M ≤ 10
        @test err < max(10.0^(-1.16 * M) * 1.05, err_min_kb)
    elseif σ ≈ 2.0
        err_max_kb = max(6 * 10.0^(-1.9 * M), 4e-14)  # error "plateaus" at ~2e-14 for M ≥ 8
        @test err < err_max_kb
    end
    nothing
end

function check_nufft_error(::Type{Float64}, ::BackwardsKaiserBesselKernel, ::HalfSupport{M}, σ, err) where {M}
    if σ ≈ 1.25
        err_min_kb = 4e-12  # error reaches a minimum at ~2e-12 for M = 10
        @test err < max(10.0^(-1.20 * M), err_min_kb)
    elseif σ ≈ 2.0
        err_max_kb = max(6 * 10.0^(-1.9 * M), 4e-14)  # error "plateaus" at ~2e-14 for M ≥ 8
        @test err < err_max_kb
    end
    nothing
end

function check_nufft_error(::Type{Float64}, ::GaussianKernel, ::HalfSupport{M}, σ, err) where {M}
    if σ ≈ 2.0
        @test err < 10.0^(-0.95 * M) * 0.8
    end
    nothing
end

function check_nufft_error(::Type{Float64}, ::BSplineKernel, ::HalfSupport{M}, σ, err) where {M}
    if σ ≈ 2.0
        @test err < 10.0^(-0.98 * M) * 0.4
    end
    nothing
end

check_nufft_error(::Type{ComplexF64}, args...) = check_nufft_error(Float64, args...)

function l2_error(us, vs)
    err = sum(zip(us, vs)) do (u, v)
        abs2(u - v)
    end
    norm = sum(abs2, vs)
    sqrt(err / norm)
end

function test_nufft_type1_1d(
        ::Type{T};
        kernel = KaiserBesselKernel(),
        N = 256,
        Np = 2 * N,
        m = HalfSupport(8),
        σ = 1.25,
    ) where {T <: Number}
    if T <: Real
        Tr = T
        ks = rfftfreq(N, Tr(N))  # wavenumbers (= [0, 1, 2, ..., N÷2])
    elseif T <: Complex
        Tr = real(T)  # real type
        ks = fftfreq(N, Tr(N))
    end

    # Generate some non-uniform random data
    rng = Random.Xoshiro(42)
    xp = rand(rng, Tr, Np) .* 2π  # non-uniform points in [0, 2π]
    vp = randn(rng, T, Np)        # random values at points

    for i ∈ eachindex(xp)
        δ = rand(rng, (-1, 0, 1))
        xp[i] += δ * 2π  # allow points outside of main unit cell
    end

    # Compute "exact" non-uniform transform
    ûs_exact = zeros(Complex{Tr}, length(ks))
    for (i, k) ∈ pairs(ks)
        ûs_exact[i] = sum(zip(xp, vp)) do (x, v)
            v * cis(-k * x)
        end
    end

    # Compute NUFFT
    ûs = Array{Complex{Tr}}(undef, length(ks))
    plan_nufft = @inferred PlanNUFFT(T, N, m; σ, kernel)
    NonuniformFFTs.set_points!(plan_nufft, xp)
    NonuniformFFTs.exec_type1!(ûs, plan_nufft, vp)

    # Check results
    err = l2_error(ûs, ûs_exact)

    # Inference tests
    if VERSION < v"1.10-"
        # On Julia 1.9, there seems to be a runtime dispatch related to throwing a
        # DimensionMismatch error using LazyStrings (in NonuniformFFTs.check_nufft_uniform_data).
        JET.@test_opt ignored_modules=(Base,) NonuniformFFTs.set_points!(plan_nufft, xp)
        JET.@test_opt ignored_modules=(Base,) NonuniformFFTs.exec_type1!(ûs, plan_nufft, vp)
    else
        JET.@test_opt NonuniformFFTs.set_points!(plan_nufft, xp)
        JET.@test_opt NonuniformFFTs.exec_type1!(ûs, plan_nufft, vp)
    end

    check_nufft_error(T, kernel, m, σ, err)

    err
end

function test_nufft_type2_1d(
        ::Type{T};
        kernel = KaiserBesselKernel(),
        N = 256,
        Np = 2 * N,
        m = HalfSupport(8),
        σ = 1.25,
    ) where {T <: Number}
    if T <: Real
        Tr = T
        ks = rfftfreq(N, Tr(N))  # wavenumbers (= [0, 1, 2, ..., N÷2])
    elseif T <: Complex
        Tr = real(T)  # real type
        ks = fftfreq(N, Tr(N))
    end

    # Generate some uniform random data + non-uniform points
    rng = Random.Xoshiro(42)
    ûs = randn(rng, Complex{Tr}, length(ks))
    xp = rand(rng, Tr, Np) .* 2π  # non-uniform points in [0, 2π]

    for i ∈ eachindex(xp)
        δ = rand(rng, (-1, 0, 1))
        xp[i] += δ * 2π  # allow points outside of main unit cell
    end

    # Compute "exact" type-2 transform (interpolation)
    vp_exact = zeros(T, Np)
    for (i, x) ∈ pairs(xp)
        for (û, k) ∈ zip(ûs, ks)
            if T <: Real
                # Complex-to-real transform with Hermitian symmetry.
                factor = ifelse(iszero(k), 1, 2)
                s, c = sincos(k * x)
                ur, ui = real(û), imag(û)
                vp_exact[i] += factor * (c * ur - s * ui)
            else
                # Usual complex-to-complex transform.
                vp_exact[i] += û * cis(k * x)
            end
        end
    end

    # Compute NUFFT
    vp = Array{T}(undef, Np)
    plan_nufft = @inferred PlanNUFFT(T, N, m; σ, kernel)
    NonuniformFFTs.set_points!(plan_nufft, xp)
    NonuniformFFTs.exec_type2!(vp, plan_nufft, ûs)

    err = l2_error(vp, vp_exact)

    # Inference tests
    if VERSION < v"1.10-"
        # On Julia 1.9, there seems to be a runtime dispatch related to throwing a
        # DimensionMismatch error using LazyStrings (in NonuniformFFTs.check_nufft_uniform_data).
        JET.@test_opt ignored_modules=(Base,) NonuniformFFTs.set_points!(plan_nufft, xp)
        JET.@test_opt ignored_modules=(Base,) NonuniformFFTs.exec_type2!(vp, plan_nufft, ûs)
    else
        JET.@test_opt NonuniformFFTs.set_points!(plan_nufft, xp)
        JET.@test_opt NonuniformFFTs.exec_type2!(vp, plan_nufft, ûs)
    end

    check_nufft_error(T, kernel, m, σ, err)

    err
end

@testset "1D NUFFTs: $T" for T ∈ (Float64, ComplexF64)
    @testset "Type 1 NUFFTs" begin
        for M ∈ 4:10
            m = HalfSupport(M)
            σ = 1.25
            @testset "$kernel (m = $M, σ = $σ)" for kernel ∈ (KaiserBesselKernel(), BackwardsKaiserBesselKernel())
                test_nufft_type1_1d(T; m, σ, kernel)
            end
            σ = 2.0
            @testset "$kernel (m = $M, σ = $σ)" for kernel ∈ (KaiserBesselKernel(), BackwardsKaiserBesselKernel(), GaussianKernel(), BSplineKernel())
                test_nufft_type1_1d(T; m, σ, kernel)
            end
        end
    end
    @testset "Type 2 NUFFTs" begin
        for M ∈ 4:10
            m = HalfSupport(M)
            σ = 1.25
            @testset "$kernel (m = $M, σ = $σ)" for kernel ∈ (KaiserBesselKernel(), BackwardsKaiserBesselKernel())
                test_nufft_type2_1d(T; m, σ, kernel)
            end
            σ = 2.0
            @testset "$kernel (m = $M, σ = $σ)" for kernel ∈ (KaiserBesselKernel(), BackwardsKaiserBesselKernel(), GaussianKernel(), BSplineKernel())
                test_nufft_type2_1d(T; m, σ, kernel)
            end
        end
    end
end