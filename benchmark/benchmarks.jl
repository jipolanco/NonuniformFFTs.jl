# Define BenchmarkTools suite for CI
# https://discourse.julialang.org/t/easy-github-benchmarking-with-new-airspeedvelocity-jl/129327

using NonuniformFFTs
using FFTW: FFTW
using Random: Xoshiro
using BenchmarkTools
using OpenCL, pocl_jll
using Adapt: adapt
using KernelAbstractions: KernelAbstractions as KA

using ThreadPinning
pinthreads(:cores)
threadinfo()

# Print OpenCL information
OpenCL.versioninfo()
@show cl.platform()
@show cl.device()

function setup_benchmark(::Type{Z}, Ns::Dims, Np; backend = CPU(), σ, m, fftw_flags = FFTW.ESTIMATE, kws...) where {Z}
    D = length(Ns)
    T = real(Z)
    rng = Xoshiro(42)
    xp_cpu = ntuple(D) do _
        randn(rng, T, Np)
    end
    vp_cpu = randn(rng, Z, Np)
    xp = adapt(backend, xp_cpu)
    vp = adapt(backend, vp_cpu)
    p = PlanNUFFT(Z, Ns; backend, σ, m, fftw_flags, kws...)
    ûs_cpu = randn(rng, complex(T), size(p))
    ûs = adapt(backend, ûs_cpu)
    (; backend, p, xp, vp, ûs, vp_out = similar(vp), ûs_out = similar(ûs))
end

function run_type1(data)
    (; backend, p, xp, vp, ûs_out,) = data
    set_points!(p, xp)
    exec_type1!(ûs_out, p, vp)
    KA.synchronize(backend)
    ûs_out
end

function run_type2(data)
    (; backend, p, xp, vp_out, ûs,) = data
    set_points!(p, xp)
    exec_type2!(vp_out, p, ûs)
    KA.synchronize(backend)
    vp_out
end

get_subsuite(suite, ::Type{Z}; backend, Ns, ρ, σ, m) where {Z} = suite[string(typeof(backend))][string(Z)]["Ns = $Ns"]["ρ = $ρ"]["σ = $σ"]["m = $m"]

## Define benchmarks

backends = [
    CPU(),
    OpenCLBackend(),
]

Ns = (128, 128, 128)
Np = prod(Ns)
σ = 1.5
m = HalfSupport(4)

ρ = Np / prod(Ns)

# Create benchmark suite for AirspeedVelocity.jl (benchmarks on github CI)
const SUITE = BenchmarkGroup()

for backend in backends, Z in (Float64, ComplexF64)
    sub = get_subsuite(SUITE, Z; backend, Ns, ρ, σ, m)
    if backend isa CPU
        for (key_atomics, use_atomics) in ("atomics" => true, "no atomics" => false)
            local data = setup_benchmark(Z, Ns, Np; backend, σ, m, use_atomics)
            sub[key_atomics]["Type 1"] = @benchmarkable run_type1($data)
            sub[key_atomics]["Type 2"] = @benchmarkable run_type2($data)
        end
    else
        for gpu_method in (:global_memory, :shared_memory)
            local data = setup_benchmark(Z, Ns, Np; backend, σ, m, gpu_method)
            sub[gpu_method]["Type 1"] = @benchmarkable run_type1($data)
            sub[gpu_method]["Type 2"] = @benchmarkable run_type2($data)
        end
    end
end

# Check that CPU and OpenCL backends give the same results.
data = setup_benchmark(Float64, Ns, Np; backend = CPU(), σ, m)
us_cpu = run_type1(data)
vp_cpu = run_type2(data)

data = setup_benchmark(Float64, Ns, Np; backend = OpenCLBackend(), σ, m)
us_opencl = run_type1(data)
vp_opencl = run_type2(data)

@show isapprox(us_cpu, Array(us_opencl); rtol = 1e-14)
@show isapprox(vp_cpu, Array(vp_opencl); rtol = 1e-14)

# Example interactive usage:
# @time tune!(SUITE)
# results_cpu = run(get_subsuite(SUITE, Float64; backend = CPU(), Ns, ρ, σ, m))
# results_opencl = run(get_subsuite(SUITE, Float64; backend = OpenCLBackend(), Ns, ρ, σ, m))
