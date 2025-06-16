using NonuniformFFTs
using AbstractFFTs: fftfreq, rfftfreq
using Random
using Test

@testset "Callbacks: $Z" for Z ∈ (Float32, ComplexF32)
    Ns = (64, 32, 16)
    Np = prod(Ns) ÷ 3
    ks = ntuple(3) do d
        f = (d == 1 && Z <: Real) ? rfftfreq : fftfreq
        f(Ns[d], Ns[d])
    end

    T = real(Z)
    rng = Xoshiro(42)
    weights = rand(rng, T, Np)
    callbacks = NUFFTCallbacks(
        nonuniform = (v, n) -> oftype(v, @inbounds(v .* weights[n])),
        uniform = (w, idx) -> let
            k⃗ = @inbounds getindex.(ks, idx)
            k² = sum(abs2, k⃗)
            factor = ifelse(iszero(k²), zero(k²), inv(k²))  # divide by k² but avoiding division by zero
            oftype(w, w .* factor)
        end,
    )

    xp = map(_ -> rand(rng, T, Np) .* T(2π), Ns)
    vp = randn(rng, Z, Np)

    # Generate reference results by applying callback functions before and after the
    # transforms (instead of passing them to exec_* functions).
    plan_base = PlanNUFFT(Z, Ns)  # plan with default parameters
    set_points!(plan_base, xp)
    reference = let plan = plan_base, type1_input = callbacks.nonuniform.(vp, eachindex(vp))
        type1_output = similar(type1_input, complex(Z), size(plan))
        exec_type1!(type1_output, plan, type1_input)
        type1_output .= callbacks.uniform.(type1_output, Tuple.(CartesianIndices(type1_output)))
        type2_input = callbacks.uniform.(type1_output, Tuple.(CartesianIndices(type1_output)))  # we apply again the callback (since we defined the same callbacks for type 1 and 2)
        type2_output = similar(type1_input)
        exec_type2!(type2_output, plan, type2_input)
        type2_output .= callbacks.nonuniform.(type2_output, eachindex(type2_output))
        (; type1_output, type2_output,)
    end

    # Test transforms with callbacks.
    # Note: we also test the non-blocking implementation (block_size = nothing)
    @testset "Block size: $block_size" for block_size in (nothing, (8, 8, 8))
        plan = PlanNUFFT(Z, Ns; block_size)
        set_points!(plan, xp)
        ws = similar(vp, complex(Z), size(plan))
        exec_type1!(ws, plan, vp; callbacks)
        @test ws ≈ reference.type1_output
        wp = similar(vp)
        exec_type2!(wp, plan, ws; callbacks)
        @test wp ≈ reference.type2_output
    end
end
