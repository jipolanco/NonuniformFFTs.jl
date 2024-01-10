using Test
using Random: Random
using AbstractFFTs: fftfreq, rfftfreq
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
        block_size = NonuniformFFTs.default_block_size(),
        sort_points = False(),
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
    plan_nufft = @inferred PlanNUFFT(T, Ns, m; σ, kernel, block_size, sort_points)
    NonuniformFFTs.set_points!(plan_nufft, xp)
    NonuniformFFTs.exec_type1!(ûs, plan_nufft, vp)

    # Check results
    err = l2_error(ûs, ûs_exact)

    check_nufft_error(T, kernel, m, σ, err)

    err
end

function test_nufft_type2(
        ::Type{T}, Ns::Dims;
        kernel = BackwardsKaiserBesselKernel(),
        Np = 2 * first(Ns),
        m = HalfSupport(8),
        σ = 1.25,
        block_size = NonuniformFFTs.default_block_size(),
        sort_points = False(),
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
    plan_nufft = @inferred PlanNUFFT(T, Ns, m; σ, kernel, block_size, sort_points)
    NonuniformFFTs.set_points!(plan_nufft, xp)
    NonuniformFFTs.exec_type2!(vp, plan_nufft, ûs)

    err = l2_error(vp, vp_exact)

    check_nufft_error(T, kernel, m, σ, err)

    err
end

@testset "2D NUFFTs: $T" for T ∈ (Float64, ComplexF64)
    Ns = (64, 64)
    block_size = 32  # small just to make sure that there are several blocks in the domain
    @testset "Type 1 NUFFTs" begin
        for M ∈ 4:8  # for σ = 1.25, going beyond M = 8 gives no improvements
            m = HalfSupport(M)
            σ = 1.25
            @testset "$kernel (m = $M, σ = $σ)" for kernel ∈ (BackwardsKaiserBesselKernel(),)
                test_nufft_type1(T, Ns; m, σ, kernel, block_size)
            end
        end
    end
    @testset "Type 2 NUFFTs" begin
        for M ∈ 4:8  # for σ = 1.25, going beyond M = 8 gives no improvements
            m = HalfSupport(M)
            σ = 1.25
            @testset "$kernel (m = $M, σ = $σ)" for kernel ∈ (BackwardsKaiserBesselKernel(),)
                test_nufft_type2(T, Ns; m, σ, kernel, block_size)
            end
        end
    end
    @testset "Blocking disabled" begin
        @testset "Type 1 NUFFT" test_nufft_type1(T, Ns; block_size = nothing)
        @testset "Type 2 NUFFT" test_nufft_type2(T, Ns; block_size = nothing)
    end
    @testset "Sort points" begin
        @testset "Type 1 NUFFT" test_nufft_type1(T, Ns; sort_points = True())
        @testset "Type 2 NUFFT" test_nufft_type2(T, Ns; sort_points = True())
    end
    @testset "Non-multiple of block size" begin
        # This gives an oversampled grid size Ñs = (80, 80) (when σ = 2.0).
        # For a block_size = 128, the block dimensions are Nb = (16, 8), so that Ñs[1] (80)
        # is not a multiple of Nb[1] (16).
        block_size = 128
        Ns′ = (37, 37)
        σ = 2.0
        @testset "Type 1 NUFFT" test_nufft_type1(T, Ns′; σ, block_size)
        @testset "Type 2 NUFFT" test_nufft_type2(T, Ns′; σ, block_size)
    end
end
