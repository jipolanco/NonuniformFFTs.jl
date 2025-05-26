# Define BenchmarkTools suite for CI
# https://discourse.julialang.org/t/easy-github-benchmarking-with-new-airspeedvelocity-jl/129327

using NonuniformFFTs
using FFTW: FFTW
using Random: Xoshiro
using BenchmarkTools

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
    p = PlanNUFFT(Z, Ns; backend, σ, m, gpu_method, fftw_flags, kws...)
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

let Ns = (128, 128, 128), Np = prod(Ns), σ = 1.5, m = HalfSupport(4)
    local ρ = Np / prod(Ns)
    local key = "CPU: Ns = $Ns, Np = $Np (density $ρ), σ = $σ, m = $m"
    for T in (Float64, ComplexF64)
        SUITE[key]["$T Type 1"] = @benchmarkable run_cpu_benchmark_type1(data) setup=(data = setup_cpu_benchmark(T, Ns, Np; σ, m))
        SUITE[key]["$T Type 2"] = @benchmarkable run_cpu_benchmark_type2(data) setup=(data = setup_cpu_benchmark(T, Ns, Np; σ, m))
    end
end
