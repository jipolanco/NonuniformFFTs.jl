using NonuniformFFTs
using FINUFFT
using Adapt
using CUDA
using FFTW
using LinearAlgebra: norm
using KernelAbstractions
using KernelAbstractions: KernelAbstractions as KA
using ThreadPinning
using Random: randn!, Xoshiro
using BenchmarkTools
using Statistics
using TimerOutputs

# NOTE: FINUFFT doesn't play well with ThreadPinning, so pinthreads should *not* be used
# for FINUFFT (CPU) benchmarks! Seems to be a conflict between ThreadPinning and OpenMP.
if haskey(ENV, "SLURM_JOB_ID")
    pinthreads(:affinitymask)
else
    pinthreads(:cores)
end

# Useful for FINUFFT (CPU):
ENV["OMP_PROC_BIND"] = "true"
# ENV["OMP_PLACES"] = "cores"
ENV["OMP_NUM_THREADS"] = Threads.nthreads()

get_device_name(::CPU) = string(Sys.cpu_info()[1].model, " ($(Sys.CPU_NAME), $(Sys.CPU_THREADS) threads)")

function get_device_name(::CUDABackend)
    dev = CUDA.device()
    io = IOBuffer()
    show(io, MIME"text/plain"(), dev)  # e.g. "CuDevice(1): NVIDIA A100 80GB PCIe"
    s = String(take!(io))
    split(s, " "; limit = 2)[2]
end

function bench_nonuniformffts(
        ::Type{Z}, backend::KA.Backend, Ns::Dims, Np;
        σ, m, gpu_method = :global_memory, kws...,
    ) where {Z}
    D = length(Ns)
    T = real(Z)
    rng = Xoshiro(42)
    xp = ntuple(D) do _
        adapt(backend, randn(rng, T, Np))
    end
    vp = adapt(backend, randn(rng, Z, Np))
    wp = similar(vp)

    p = PlanNUFFT(Z, Ns; backend, σ, m, gpu_method, kws...)
    ûs = similar(vp, complex(T), size(p))

    # Execute plan once
    set_points!(p, xp)
    exec_type1!(ûs, p, vp)
    exec_type2!(wp, p, ûs)

    # Compare with reference case (high accuracy: rtol = 1e-14)
    relative_errors = let p_ref = PlanNUFFT(Z, Ns; backend, kws..., gpu_method = :global_memory, m = HalfSupport(8), σ = T(2.0))
        ûs_ref = similar(ûs)
        wp_ref = similar(wp)
        set_points!(p_ref, xp)
        exec_type1!(ûs_ref, p_ref, vp)
        exec_type2!(wp_ref, p_ref, ûs_ref)
        local type1 = norm(ûs_ref - ûs) / norm(ûs_ref)
        local type2 = norm(wp_ref - wp) / norm(wp_ref)
        KA.unsafe_free!(ûs_ref)
        KA.unsafe_free!(wp_ref)
        (; type1, type2,)
    end

    if backend isa KA.CPU
        reset_timer!(p.timer)
    end

    type1 = @benchmark let
        set_points!($p, $xp)
        exec_type1!($ûs, $p, $vp)
        KA.synchronize($backend)
    end

    type2 = @benchmark let
        set_points!($p, $xp)
        exec_type2!($wp, $p, $ûs)
        KA.synchronize($backend)
    end

    if backend isa KA.CPU
        println(p.timer)
    end

    (; type1, type2, relative_errors,)
end

function bench_finufft_cpu(::Type{Z}, Ns::Dims, Np; tol, opts...) where {Z <: Complex}
    backend = CPU()
    T = real(Z)
    D = length(Ns)
    rng = Xoshiro(42)
    xp = ntuple(D) do _
        adapt(backend, randn(rng, T, Np))
    end
    vp = adapt(backend, randn(rng, Z, Np))
    ûs = similar(vp, complex(T), Ns)
    ntrans = 1
    n_modes = collect(Ns)
    plan_type1 = finufft_makeplan(1, n_modes, -1, ntrans, tol; dtype = T, opts...)
    finalizer(finufft_destroy!, plan_type1)
    type1 = @benchmark let
        finufft_setpts!($plan_type1, $xp...)
        finufft_exec!($plan_type1, $vp, $ûs)
        KA.synchronize($backend)
    end
    (; type1,)
end

function bench_cufinufft(::Type{Z}, Ns::Dims, Np; tol, opts...) where {Z <: Complex}
    backend = CUDABackend()
    T = real(Z)
    D = length(Ns)
    rng = Xoshiro(42)
    xp = ntuple(D) do _
        adapt(backend, randn(rng, T, Np))
    end
    vp = adapt(backend, randn(rng, Z, Np))
    wp = similar(vp)
    ûs = similar(vp, complex(T), Ns)
    ntrans = 1
    n_modes = collect(Ns)

    plan_type1 = cufinufft_makeplan(1, n_modes, -1, ntrans, tol; dtype = T, opts...)
    plan_type2 = cufinufft_makeplan(2, n_modes, +1, ntrans, tol; dtype = T, opts...)
    finalizer(cufinufft_destroy!, plan_type1)
    finalizer(cufinufft_destroy!, plan_type2)

    # Execute plans once
    cufinufft_setpts!(plan_type1, xp...)
    cufinufft_setpts!(plan_type2, xp...)
    cufinufft_exec!(plan_type1, vp, ûs)
    cufinufft_exec!(plan_type2, ûs, wp)

    # Compare with reference case (high accuracy: rtol = 1e-14)
    relative_errors = let tol = 1e-14
        p1_ref = cufinufft_makeplan(1, n_modes, -1, ntrans, tol; dtype = T, opts..., gpu_method = 1, upsampfac = 2.0)
        p2_ref = cufinufft_makeplan(2, n_modes, +1, ntrans, tol; dtype = T, opts..., gpu_method = 1, upsampfac = 2.0)

        ûs_ref = similar(ûs)
        wp_ref = similar(wp)

        cufinufft_setpts!(p1_ref, xp...)
        cufinufft_setpts!(p2_ref, xp...)
        cufinufft_exec!(p1_ref, vp, ûs_ref)
        cufinufft_exec!(p2_ref, ûs_ref, wp_ref)

        cufinufft_destroy!(p1_ref)
        cufinufft_destroy!(p2_ref)

        local type1 = norm(ûs_ref - ûs) / norm(ûs_ref)
        local type2 = norm(wp_ref - wp) / norm(wp_ref)

        KA.unsafe_free!(ûs_ref)
        KA.unsafe_free!(wp_ref)

        (; type1, type2,)
    end

    type1 = @benchmark let
        cufinufft_setpts!($plan_type1, $xp...)
        cufinufft_exec!($plan_type1, $vp, $ûs)
        KA.synchronize($backend)
    end

    type2 = @benchmark let
        cufinufft_setpts!($plan_type2, $xp...)
        cufinufft_exec!($plan_type2, $ûs, $wp)
        KA.synchronize($backend)
    end

    (; type1, type2, relative_errors,)
end

function bench_finufft_cpu(::Type{Z}, Ns::Dims, Np; tol, opts...) where {Z <: Complex}
    T = real(Z)
    D = length(Ns)
    rng = Xoshiro(42)
    xp = ntuple(D) do _
        randn(rng, T, Np)
    end
    vp = randn(rng, Z, Np)
    wp = similar(vp)
    ûs = similar(vp, complex(T), Ns)
    ntrans = 1
    n_modes = collect(Ns)

    plan_type1 = finufft_makeplan(1, n_modes, -1, ntrans, tol; dtype = T, opts...)
    plan_type2 = finufft_makeplan(2, n_modes, +1, ntrans, tol; dtype = T, opts...)
    finalizer(finufft_destroy!, plan_type1)
    finalizer(finufft_destroy!, plan_type2)

    # Execute plans once
    finufft_setpts!(plan_type1, xp...)
    finufft_setpts!(plan_type2, xp...)
    finufft_exec!(plan_type1, vp, ûs)
    finufft_exec!(plan_type2, ûs, wp)

    # Compare with reference case (high accuracy: rtol = 1e-14)
    relative_errors = let tol = 1e-14
        p1_ref = finufft_makeplan(1, n_modes, -1, ntrans, tol; dtype = T, opts..., upsampfac = 2.0)
        p2_ref = finufft_makeplan(2, n_modes, +1, ntrans, tol; dtype = T, opts..., upsampfac = 2.0)

        ûs_ref = similar(ûs)
        wp_ref = similar(wp)

        finufft_setpts!(p1_ref, xp...)
        finufft_setpts!(p2_ref, xp...)
        finufft_exec!(p1_ref, vp, ûs_ref)
        finufft_exec!(p2_ref, ûs_ref, wp_ref)

        finufft_destroy!(p1_ref)
        finufft_destroy!(p2_ref)

        local type1 = norm(ûs_ref - ûs) / norm(ûs_ref)
        local type2 = norm(wp_ref - wp) / norm(wp_ref)

        (; type1, type2,)
    end

    type1 = @benchmark let
        finufft_setpts!($plan_type1, $xp...)
        finufft_exec!($plan_type1, $vp, $ûs)
    end

    type2 = @benchmark let
        finufft_setpts!($plan_type2, $xp...)
        finufft_exec!($plan_type2, $ûs, $wp)
    end

    (; type1, type2, relative_errors,)
end

function run_benchmark_nonuniformffts(
        ::Type{Z},
        backend,
        Ns::Dims,
        Nps::AbstractVector;  # number of points per test case
        σ = 1.5, m = HalfSupport(4),  # rtol = 1e-6
        gpu_method = :global_memory,
        fftw_flags = FFTW.ESTIMATE,
        params...,
    ) where {Z <: Number}
    N = first(Ns)
    suffix = if backend isa GPU
        "_$gpu_method"
    else
        ""
    end
    device_name = get_device_name(backend)
    filename = "NonuniformFFTs_$(N)_$(Z)_$(typeof(backend))$(suffix).dat"
    open(filename, "w") do io
        println(io, "# NonuniformFFTs.jl using $(typeof(backend))")
        println(io, "# Benchmark: NUFFT of scalar data")
        println(io, "#  - Backend: ", typeof(backend))
        println(io, "#  - Device: ", device_name)
        println(io, "#  - Element type: ", Z)
        println(io, "#  - Grid size: ", Ns)
        println(io, "#  - Oversampling factor: ", σ)
        println(io, "#  - Half support: ", m)
        if backend isa KA.GPU
            println(io, "#  - GPU method: ", gpu_method)
        else
            println(io, "#  - Number of threads: ", Threads.nthreads())
        end
        println(io, "# (1) Number of points  (2) Type 1 (median, s)  (3) Type 2 (median, s)  (4) Relative error type 1  (5) Relative error type 2")
        for Np ∈ Nps
            bench = bench_nonuniformffts(Z, backend, Ns, Np; params..., σ, m, gpu_method, fftw_flags)
            type1 = median(bench.type1.times) / 1e9  # in seconds
            type2 = median(bench.type2.times) / 1e9  # in seconds
            @show Np, type1, type2, bench.relative_errors.type1, bench.relative_errors.type2
            join(io, (Np, type1, type2, bench.relative_errors.type1, bench.relative_errors.type2), '\t')
            print(io, '\n')
            flush(io)
        end
    end
    filename
end

function run_benchmark_cufinufft(
        ::Type{Z},
        Ns::Dims,
        Nps::AbstractVector;  # number of points per test case
        tol = 1e-6,
        modeord = 1,     # convention used by default in NonuniformFFTs
        gpu_method = 1,  # 1: global memory (-sort) method | 2: shared memory | method 0 or 2 is much slower, at least with Float64 data
        gpu_sort = 1,    # needed for good performance
        gpu_kerevalmeth = 1,  # setting it to 0 (direct evaluation) doesn't make a big difference?
        params...,
    ) where {Z <: Complex}
    N = first(Ns)
    method = gpu_method == 1 ? :global_memory : :shared_memory
    # Make sure we use the current CUDA device and stream (otherwise KA.synchronize won't
    # work as expected).
    params_cuda = (
        gpu_device_id = CUDA.deviceid(CUDA.device()),
        gpu_stream = Base.unsafe_convert(Ptr{CUDA.CUstream_st}, CUDA.stream()),
    )
    device_name = get_device_name(CUDABackend())
    filename = "CuFINUFFT_$(N)_$(Z)_$(method).dat"
    open(filename, "w") do io
        println(io, "# CuFINUFFT")
        println(io, "# Benchmark: NUFFT of scalar data")
        println(io, "#  - Device: ", device_name)
        println(io, "#  - Element type: ", Z)
        println(io, "#  - Grid size: ", Ns)
        println(io, "#  - Relative tolerance: ", tol)
        println(io, "#  - GPU sort: ", gpu_sort)
        println(io, "#  - GPU kernel evaluation method: ", gpu_kerevalmeth)
        println(io, "#  - Order of Fourier modes (modeord): ", modeord)
        println(io, "#  - GPU method: ", gpu_method, " ($method)")
        println(io, "# (1) Number of points  (2) Type 1 (median, s)  (3) Type 2 (median, s)  (4) Relative error type 1  (5) Relative error type 2")
        for Np ∈ Nps
            bench = bench_cufinufft(Z, Ns, Np; params..., params_cuda..., tol, modeord, gpu_method, gpu_sort, gpu_kerevalmeth)
            type1 = median(bench.type1.times) / 1e9  # in seconds
            type2 = median(bench.type2.times) / 1e9  # in seconds
            @show Np, type1, type2, bench.relative_errors.type1, bench.relative_errors.type2
            join(io, (Np, type1, type2, bench.relative_errors.type1, bench.relative_errors.type2), '\t')
            print(io, '\n')
            flush(io)
        end
    end
    filename
end

function run_benchmark_finufft_cpu(
        ::Type{Z},
        Ns::Dims,
        Nps::AbstractVector;  # number of points per test case
        tol = 1e-6,
        modeord = 1,     # convention used by default in NonuniformFFTs
        spread_sort = 1,    # needed for good performance
        spread_kerevalmeth = 1,
        fftw = FFTW.ESTIMATE,
        nthreads = Threads.nthreads(),
        params...,
    ) where {Z <: Complex}
    N = first(Ns)
    device_name = get_device_name(CPU())
    filename = "FINUFFT_$(N)_$(Z)_CPU.dat"
    open(filename, "w") do io
        println(io, "# FINUFFT (CPU)")
        println(io, "# Benchmark: NUFFT of scalar data")
        println(io, "#  - Device: ", device_name)
        println(io, "#  - Number of threads: ", nthreads)
        println(io, "#  - Element type: ", Z)
        println(io, "#  - Grid size: ", Ns)
        println(io, "#  - Relative tolerance: ", tol)
        println(io, "#  - Sort (spread_sort): ", spread_sort)
        println(io, "#  - Kernel evaluation method (spread_kerevalmeth): ", spread_kerevalmeth)
        println(io, "#  - Order of Fourier modes (modeord): ", modeord)
        println(io, "# (1) Number of points  (2) Type 1 (median, s)  (3) Type 2 (median, s)  (4) Relative error type 1  (5) Relative error type 2")
        for Np ∈ Nps
            bench = bench_finufft_cpu(Z, Ns, Np; params..., tol, fftw, nthreads, modeord, spread_sort, spread_kerevalmeth)
            type1 = median(bench.type1.times) / 1e9  # in seconds
            type2 = median(bench.type2.times) / 1e9  # in seconds
            @show Np, type1, type2, bench.relative_errors.type1, bench.relative_errors.type2
            join(io, (Np, type1, type2, bench.relative_errors.type1, bench.relative_errors.type2), '\t')
            print(io, '\n')
            flush(io)
        end
    end
    filename
end

function run_all_benchmarks()
    T = Float64
    Z = complex(T)
    Ngrid = 256
    Ns = (1, 1, 1) .* Ngrid

    # Parameters for 1e-6 accuracy
    σ = 1.5
    m = HalfSupport(4)

    ρs = exp10.(-4:0.5:1)  # point densities
    Nprod = prod(Ns)
    Nps = map(ρs) do ρ
        round(Int, ρ * Nprod)
    end

    # Real data
    run_benchmark_nonuniformffts(T, CUDABackend(), Ns, Nps; σ, m, gpu_method = :global_memory)
    run_benchmark_nonuniformffts(T, CUDABackend(), Ns, Nps; σ, m, gpu_method = :shared_memory)
    run_benchmark_nonuniformffts(T, CPU(), Ns, Nps; σ, m)

    # Complex data
    run_benchmark_nonuniformffts(Z, CUDABackend(), Ns, Nps; σ, m, gpu_method = :global_memory)
    run_benchmark_nonuniformffts(Z, CUDABackend(), Ns, Nps; σ, m, gpu_method = :shared_memory)
    run_benchmark_nonuniformffts(Z, CPU(), Ns, Nps; σ, m)

    # CuFINUFFT
    params_cufinufft = (;
        gpu_sort = 1,    # needed for good performance
        gpu_kerevalmeth = 1,  # setting it to 0 (direct evaluation) doesn't make a big difference?
        tol = 1e-6,
    )

    gpu_method = 1  # global memory (GM-sort)
    run_benchmark_cufinufft(Z, Ns, Nps; gpu_method, params_cufinufft...)

    gpu_method = 2  # shared memory (SM)
    run_benchmark_cufinufft(Z, Ns, Nps; gpu_method, params_cufinufft...)

    # FINUFFT CPU
    params_finufft = (;
        spread_sort = 1,
        spread_kerevalmeth = 1,
        tol = 1e-6,
    )
    run_benchmark_finufft_cpu(Z, Ns, Nps; params_finufft...)

    nothing
end

outdir = "results"
mkpath(outdir)
cd(outdir) do
    run_all_benchmarks()
end
