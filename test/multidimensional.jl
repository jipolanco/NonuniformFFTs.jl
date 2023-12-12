using Test
using Random: Random
using AbstractFFTs: fftfreq, rfftfreq
using JET: JET
using StaticArrays: SVector
using LinearAlgebra: ⋅
using NonuniformFFTs

function check_nufft_error(::Type{Float64}, ::BackwardsKaiserBesselKernel, ::HalfSupport{M}, σ, err) where {M}
    if σ ≈ 1.25
        err_min_kb = 4e-12  # error reaches a minimum at ~2e-12 for M = 10
        @test err < max(10.0^(-1.20 * M) * 2, err_min_kb)
    elseif σ ≈ 2.0
        err_max_kb = max(6 * 10.0^(-1.9 * M), 4e-14)  # error "plateaus" at ~2e-14 for M ≥ 8
        @test err < err_max_kb
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

function test_nufft_type1(
        ::Type{T}, Ns::Dims;
        kernel = BackwardsKaiserBesselKernel(),
        Np = 2 * first(Ns),
        m = HalfSupport(8),
        σ = 1.25,
    ) where {T <: Number}
    Tr = real(T)
    ks = map(N -> fftfreq(N, Tr(N)), Ns)
    if T <: Real
        ks = Base.setindex(ks, rfftfreq(Ns[1], Tr(Ns[1])), 1)  # we perform r2c transform along first dimension
    end

    # Generate some non-uniform random data
    rng = Random.Xoshiro(42)
    d = length(Ns)
    xp = rand(rng, SVector{d, Tr}, Np)  # non-uniform points in [0, 1]ᵈ
    for i ∈ eachindex(xp)
        xp[i] = xp[i] .* 2π  # rescale points to [0, 2π]ᵈ
    end
    vp = randn(rng, T, Np)   # random values at points

    # Compute "exact" non-uniform transform
    ûs_exact = zeros(Complex{Tr}, map(length, ks))
    for I ∈ CartesianIndices(ûs_exact)
        k⃗ = SVector(map(getindex, ks, Tuple(I)))
        for (x⃗, v) ∈ zip(xp, vp)
            ûs_exact[I] += v * cis(-k⃗ ⋅ x⃗)
        end
    end

    # Compute NUFFT
    ûs = Array{Complex{Tr}}(undef, map(length, ks))
    plan_nufft = @inferred PlanNUFFT(T, Ns, m; σ, kernel)
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

function test_nufft_type2(
        ::Type{T}, Ns::Dims;
        kernel = BackwardsKaiserBesselKernel(),
        Np = 2 * first(Ns),
        m = HalfSupport(8),
        σ = 1.25,
    ) where {T <: Number}
    Tr = real(T)
    ks = map(N -> fftfreq(N, Tr(N)), Ns)
    if T <: Real
        ks = Base.setindex(ks, rfftfreq(Ns[1], Tr(Ns[1])), 1)  # we perform r2c transform along first dimension
    end

    # Generate some uniform random data + non-uniform points
    rng = Random.Xoshiro(42)
    ûs = randn(rng, Complex{Tr}, map(length, ks))
    d = length(Ns)
    xp = rand(rng, SVector{d, Tr}, Np)  # non-uniform points in [0, 1]ᵈ
    for i ∈ eachindex(xp)
        xp[i] = xp[i] .* 2π  # rescale points to [0, 2π]ᵈ
    end

    # Compute "exact" type-2 transform (interpolation)
    vp_exact = zeros(T, Np)
    for I ∈ CartesianIndices(ûs)
        k⃗ = SVector(map(getindex, ks, Tuple(I)))
        û = ûs[I]
        for i ∈ eachindex(xp, vp_exact)
            x⃗ = xp[i]
            if T <: Real
                # Complex-to-real transform with Hermitian symmetry.
                factor = ifelse(iszero(k⃗[1]), 1, 2)
                s, c = sincos(k⃗ ⋅ x⃗)
                ur, ui = real(û), imag(û)
                vp_exact[i] += factor * (c * ur - s * ui)
            else
                # Usual complex-to-complex transform.
                vp_exact[i] += û * cis(k⃗ ⋅ x⃗)
            end
        end
    end

    # Compute NUFFT
    vp = Array{T}(undef, Np)
    plan_nufft = @inferred PlanNUFFT(T, Ns, m; σ, kernel)
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

@testset "2D NUFFTs: $T" for T ∈ (Float64, ComplexF64)
    Ns = (64, 64)
    @testset "Type 1 NUFFTs" begin
        for M ∈ 4:8  # for σ = 1.25, going beyond M = 8 gives no improvements
            m = HalfSupport(M)
            σ = 1.25
            @testset "$kernel (m = $M, σ = $σ)" for kernel ∈ (BackwardsKaiserBesselKernel(),)
                test_nufft_type1(T, Ns; m, σ, kernel)
            end
        end
    end
    @testset "Type 2 NUFFTs" begin
        for M ∈ 4:8  # for σ = 1.25, going beyond M = 8 gives no improvements
            m = HalfSupport(M)
            σ = 1.25
            @testset "$kernel (m = $M, σ = $σ)" for kernel ∈ (BackwardsKaiserBesselKernel(),)
                test_nufft_type2(T, Ns; m, σ, kernel)
            end
        end
    end
end
