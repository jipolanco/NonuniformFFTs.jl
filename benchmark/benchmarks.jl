# Define BenchmarkTools suite for CI
# https://discourse.julialang.org/t/easy-github-benchmarking-with-new-airspeedvelocity-jl/129327

using NonuniformFFTs
using FFTW: FFTW
using Random: Xoshiro
using BenchmarkTools

using ThreadPinning
pinthreads(:cores)

# Create benchmark suite for AirspeedVelocity.jl (benchmarks on github CI)
const SUITE = BenchmarkGroup()

function setup_cpu_benchmark(::Type{Z}, Ns::Dims, Np; σ, m, fftw_flags = FFTW.ESTIMATE, kws...) where {Z}
    D = length(Ns)
    T = real(Z)
    rng = Xoshiro(42)
    xp = ntuple(D) do _
        randn(rng, T, Np)
    end
    vp = randn(rng, Z, Np)
    backend = CPU()
    p = PlanNUFFT(Z, Ns; backend, σ, m, fftw_flags, kws...)
    ûs = similar(vp, complex(T), size(p))
    (; p, xp, vp, ûs,)
end

function run_cpu_benchmark_type1(data)
    (; p, xp, vp, ûs,) = data
    set_points!(p, xp)
    exec_type1!(ûs, p, vp)
    nothing
end

function run_cpu_benchmark_type2(data)
    (; p, xp, vp, ûs,) = data
    set_points!(p, xp)
    exec_type2!(vp, p, ûs)
    nothing
end

## Run benchmarks

Ns = (128, 128, 128)
Np = prod(Ns)
σ = 1.5
m = HalfSupport(4)

ρ = Np / prod(Ns)
case = "CPU: Ns = $Ns, ρ = $ρ, σ = $σ, m = $m"

for Z in (Float64, ComplexF64)
    for (key_atomics, use_atomics) in ("atomics" => true, "no atomics" => false)
        local data = setup_cpu_benchmark(Z, Ns, Np; σ, m, use_atomics)
        SUITE[case][key_atomics]["$Z Type 1"] = @benchmarkable run_cpu_benchmark_type1($data)
        SUITE[case][key_atomics]["$Z Type 2"] = @benchmarkable run_cpu_benchmark_type2($data)
    end
end

# Example interactive usage:
# @time tune!(SUITE)
# results = run(SUITE[case]["atomics"]["Float64 Type 1"]; verbose = true, seconds = 2)
# results = run(SUITE[case]["atomics"]["ComplexF64 Type 1"]; verbose = true, seconds = 2)
# results = run(SUITE[case]["no atomics"]["Float64 Type 1"]; verbose = true, seconds = 2)
# results = run(SUITE[case]["no atomics"]["ComplexF64 Type 1"]; verbose = true, seconds = 2)
