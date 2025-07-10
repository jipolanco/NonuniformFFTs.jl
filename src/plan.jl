abstract type AbstractNUFFTData{T <: Number, N, Nc} end

struct RealNUFFTData{
        T <: AbstractFloat, N, Nc,
        WaveNumbers <: NTuple{N, AbstractVector{T}},
        FieldsR <: NTuple{Nc, AbstractArray{T, N}},
        FieldsC <: NTuple{Nc, AbstractArray{Complex{T}, N}},
        PlanFFT_fw <: AbstractFFTs.Plan{T},  # this requires CUDA.jl 5.5.2 (see CUDA.jl#2504)
        PlanFFT_bw <: AbstractFFTs.Plan{Complex{T}},
    } <: AbstractNUFFTData{T, N, Nc}
    ks      :: WaveNumbers  # wavenumbers in *non-oversampled* Fourier grid
    us      :: FieldsR      # values in oversampled grid (real)
    ûs      :: FieldsC      # Fourier coefficients in oversampled grid (complex)
    plan_fw :: PlanFFT_fw
    plan_bw :: PlanFFT_bw
end

struct ComplexNUFFTData{
        T <: AbstractFloat, N, Nc,
        WaveNumbers <: NTuple{N, AbstractVector{T}},
        FieldsC <: NTuple{Nc, AbstractArray{Complex{T}, N}},
        PlanFFT_fw <: AbstractFFTs.Plan{Complex{T}},
        PlanFFT_bw <: AbstractFFTs.Plan{Complex{T}},
    } <: AbstractNUFFTData{Complex{T}, N, Nc}
    ks      :: WaveNumbers
    us      :: FieldsC
    plan_fw :: PlanFFT_fw  # in-place transform
    plan_bw :: PlanFFT_bw  # inverse in-place transform
end

# Here the "input" means the gridded data in "physical" space, while the "output"
# corresponds to its Fourier coefficients.
output_field(data::RealNUFFTData) = data.ûs
output_field(data::ComplexNUFFTData) = data.us  # output === input

# Case of real data
function init_plan_data(
        ::Type{T}, backend::KA.Backend, Ñs::Dims, ks::NTuple, ::Val{Nc};
        plan_kwargs,
    ) where {T <: AbstractFloat, Nc}
    @assert Nc ≥ 1
    us = ntuple(_ -> KA.zeros(backend, T, Ñs), Val(Nc))
    dims_out = (Ñs[1] ÷ 2 + 1, Base.tail(Ñs)...)
    ûs = ntuple(_ -> KA.zeros(backend, Complex{T}, dims_out), Val(Nc))
    plan_fw = AbstractFFTs.plan_rfft(first(us); plan_kwargs...)
    plan_bw = AbstractFFTs.plan_brfft(first(ûs), Ñs[1]; plan_kwargs...)
    RealNUFFTData(ks, us, ûs, plan_fw, plan_bw)
end

# Case of complex data
function init_plan_data(
        ::Type{Complex{T}}, backend::KA.Backend, Ñs::Dims, ks::NTuple, ::Val{Nc};
        plan_kwargs,
    ) where {T <: AbstractFloat, Nc}
    @assert Nc ≥ 1
    us = ntuple(_ -> KA.zeros(backend, Complex{T}, Ñs), Val(Nc))
    plan_fw = AbstractFFTs.plan_fft!(first(us); plan_kwargs...)
    plan_bw = AbstractFFTs.plan_bfft!(first(us); plan_kwargs...)
    ComplexNUFFTData(ks, us, plan_fw, plan_bw)
end

"""
    NUFFTCallbacks(; nonuniform, uniform,)

Optional callback functions to be applied at different stages of a NUFFT.

These are user-defined functions that allow to modify "on the fly" the input and/or the
output of a NUFFT (type 1 or 2).

This can be useful for performance (as it allows to combine operations and reduce the number
of passes through memory) and for reducing memory usage (avoiding allocation of new arrays).
Note that **transform inputs are never modified by the callback**, which means that one
can safely do further operations on the original input data after the transform.

Callbacks defined via `NUFFTCallbacks` are accepted by [`exec_type1!`](@ref) and [`exec_type2!`](@ref)
via their `callbacks` keyword argument.

# Callback types

One can define two types of callback functions:

1.  a callback on **non-uniform data** (input of type-1 NUFFT / output of type-2 NUFFT).

    Its signature should be `nonuniform(v::Tuple, n::Integer)`, where `v = (v₁, v₂, …)` is the
    "original" value at the non-uniform point with index `n` (i.e. in the `points` array of [`set_points!`](@ref)).

    Note: the length of `v` is equal to the number of _transforms_ (`ntransforms` argument of [`PlanNUFFT`](@ref)).

2.  a callback on **uniform data** (output of type-1 NUFFT / input of type-2 NUFFT).

    Its signature should be `uniform(w::Tuple, idx::Tuple)`, where `w = (w₁, w₂, …)` is
    the "original" value at grid point with index `idx = (i₁, i₂, …)`. Note that `w` and `idx`
    are tuples which may have different lengths:

    - the length of `w` is equal to the number of _transforms_ (`ntransforms` argument of [`PlanNUFFT`](@ref));
    - the length of `idx` is equal to the number of _dimensions_ (e.g. 3 for 3D transforms).

!!! warning "Output type"

    One must make sure that the value returned by the callback has the **same type** as the input.
    For instance, if uniform data has type `ComplexF32`, then values returned by `uniform`
    must also be `ComplexF32`. One may use [`oftype`](https://docs.julialang.org/en/v1/base/base/#Base.oftype)
    to ensure this (see examples below).

# Examples

## Callback on non-uniform data

Define a callback function that multiplies each non-uniform point by random weights in 2D:

```julia
Np = 1000                                # number of non-uniform points
xs = (rand(Np) .* 2pi, rand(Np) .* 2pi)  # random non-uniform points in [0, 2π]²
weights = rand(Np)                       # random weights

callback_nu(v, n) = oftype(v, v .* weights[n])   # define callback which will multiply each non-uniform value by its corresponding weight
callbacks = NUFFTCallbacks(nonuniform = callback_nu)

exec_type1!(output, plan, input; callbacks = callbacks)  # use callback in type-1 transform (for example)
```

## Callback on uniform data

Define a callback function that multiplies each uniform point by ``|\\bm{k}|^2`` (where
``\\bm{k}`` can represent a Fourier wavevector):

```julia
using AbstractFFTs: fftfreq
Nx = Ny = 256                    # dimensions of uniform grid
ws = rand(ComplexF64, (Nx, Ny))  # random non-uniform data
kx = fftfreq(Nx, Nx)             # wavenumbers (frequencies) in x direction
ky = fftfreq(Ny, Ny)             # wavenumbers (frequencies) in y direction

function callback_u(w, idx)
    i, j = idx
    k² = kx[i]^2 + ky[j]^2
    oftype(w, w .* k²)
end

callbacks = NUFFTCallbacks(uniform = callback_u)

exec_type1!(output, plan, input; callbacks = callbacks)  # use callback in type-1 transform (for example)
```

"""
struct NUFFTCallbacks{
        CallbackNU <: Function,
        CallbackU <: Function,
    }
    nonuniform::CallbackNU
    uniform::CallbackU
end

function Adapt.adapt_structure(to, c::NUFFTCallbacks)
    NUFFTCallbacks(
        adapt(to, c.nonuniform),
        adapt(to, c.uniform),
    )
end

NUFFTCallbacks(; nonuniform = default_callback, uniform = default_callback) = NUFFTCallbacks(nonuniform, uniform)

# By default, the callback returns the first passed argument (which is the input or output NUFFT value).
@inline default_callback(v, args...) = v

"""
    PlanNUFFT([T = ComplexF64], dims::Dims; ntransforms = Val(1), backend = CPU(), kwargs...)

Construct a plan for performing non-uniform FFTs (NUFFTs).

The created plan contains all data needed to perform NUFFTs for non-uniform data of type `T`
(`ComplexF64` by default) and uniform data with dimensions `dims`.

# Extended help

## Optional keyword arguments

- `ntransforms = Val(1)`: the number of simultaneous transforms to perform.
  This is useful if one wants to transform multiple scalar quantities at the same
  non-uniform points.

- `backend::KernelAbstractions.Backend = CPU()`: corresponds to the device type where
  everything will be executed. This could be e.g. `CUDABackend()` if CUDA.jl is loaded.

### NUFFT parameters

The following parameters control transform accuracy. The default values give a relative accuracy of
the order of ``10^{-7}`` for `Float64` or `ComplexF64` data.

- `m = HalfSupport(4)`: the half-support of the convolution kernels. Large values
  increase accuracy at the cost of performance.

- `σ = 2.0`: NUFFT oversampling factor. Typical values are 2.0 (more accurate) and 1.25 (faster),
  but other values such as 1.5 should also work.

- `kernel::AbstractKernel = BackwardsKaiserBesselKernel()`: convolution kernel used for NUFFTs.

### Main performance parameters

- `kernel_evalmode`: method used for kernel evaluation.
  The default is [`FastApproximation`](@ref) on CPU, which will attempt to use a fast
  approximation method which greatly speeds up kernel evaluation.
  On NVIDIA GPUs the default is [`Direct`](@ref), as the fast approximation method is not
  necessarily faster.

- `block_size`: the block size (in number of elements) when using block partitioning or when
  sorting is enabled.
  This enables spatial sorting of points, even when `sort_points = False()` (which actually
  permutes point data for possibly faster memory accesses).
  The block size can be tuned for maximal performance.
  It can either be passed as an `Int` (linear block size) or as a tuple `(B₁, …, Bₙ)` to
  specify the block size in each Cartesian direction.
  The current default is 4096 on the CPU and around 1024 on the GPU (but depends on the number of dimensions).
  These may change in the future or even depend on the actual computing device.
  On the CPU, using block partitioning is required for running with multiple threads.
  On the GPU, this option is ignored if `gpu_method = :shared_memory`.
  Blocking / spatial sorting can be completely disabled by passing `block_size = nothing` (but this is
  generally slower).

- `use_atomics = false`: if `true`, atomic operations are used in type-1 transforms (spreading).
  This can improve performance when using a large number of CPU threads.
  Otherwise, if `false` (default), we use a `ReentrantLock` to make sure that only one
  thread writes at once to the output array.
  This does not affect GPU performance.

- `gpu_method = :global_memory`: allows to select between different implementations of
  GPU transforms. Possible options are:

  * `:global_memory` (default): directly read and write onto arrays in global memory in spreading
    (type-1) and interpolation (type-2) operations;

  * `:shared_memory`: copy data between global memory and shared memory (local
    to each GPU workgroup) and perform most operations in the latter, which is faster and
    can help avoid some atomic operations in type-1 transforms. We try to use as much shared
    memory as is typically available on current GPUs (which is typically 48 KiB on
    CUDA and 64 KiB on AMDGPU). Note that this method completely ignores the `block_size`
    parameter, as the actual block size is adjusted to maximise shared memory usage. When
    this method is enabled, one can play with the `gpu_batch_size` parameter (see below) to
    further tune performance.

  For highly dense problems (number of non-uniform points comparable to the total grid
  size), the `:shared_memory` method can be much faster, especially when the `HalfSupport`
  is 4 or less (accuracies up to `1e-7` for `σ = 2`).

- `fftw_flags = FFTW.MEASURE`: parameters passed to the FFTW planner when `backend = CPU()`.

### Other performance parameters

These are more advanced performance parameters which may disappear or whose behaviour may
change in the future.

- `sort_points = False()`: whether to internally permute the order of the non-uniform points.
  This can be enabled by passing `sort_points = True()`.
  This will generally require extra allocations since the input points need to be copied onto a new container.
  If this is enabled, more time will be spent in [`set_points!`](@ref) and less time on the actual transforms.
  This can improve performance if executing multiple transforms on the same non-uniform points.
  Note that, even when enabled, this does not modify the `points` argument passed to `set_points!`.
  This option is ignored when `block_size = nothing` (which disables spatial sorting).

- `gpu_batch_size = Val(Np)`: minimum batch size used in type-1 transforms when `gpu_method = :shared_memory`.
  The idea is that, to avoid inefficient atomic operations on shared-memory arrays, we process
  non-uniform points in batches of `Np` points.
  The actual value of `Np` will typically be larger than the input one in order to maximise
  shared memory usage within each GPU workgroup.
  Note that larger `Np` also means that less shared memory space is available for local blocks,
  meaning that the effective block size can get smaller (which is not necessarily bad for performance,
  and can actually be beneficial).
  When tuning performance, it is helpful to print the plan (as in `println(plan)`) to see
  the actual block and batch sizes.

### Other parameters

- `fftshift = false`: determines the order of wavenumbers in uniform space.
  If `false` (default), the same order used by FFTW is used, with positive wavenumbers first
  (`[0, 1, 2, …, N÷2-1]` for even-size transforms) and negative ones afterwards (`[-N÷2, …, -1]`).
  Otherwise, wavenumbers are expected to be in increasing order (`[-N÷2, -kmax, …, -1, 0, 1, …, N÷2-1]`),
  which is compatible with the output in NFFT.jl and corresponds to applying the
  `AbstractFFTs.fftshift` function to the data.
  This option also corresponds to the `modeord` parameter in FINUFFT.
  This only affects complex-to-complex transforms.

- `timer = TimerOutput()`: allows to specify a `TimerOutput` (from the
  [TimerOutputs.jl](https://github.com/KristofferC/TimerOutputs.jl) package) where timing
  information will be written to.
  By default the plan creates its own timer.
  One can visualise the time spent on different parts of the NUFFT computation using `p.timer`.

- `synchronise = false`: if `true`, add synchronisation barrier between calls to GPU kernels.
  Enabling this is needed for accurate timings in `p.timer` when computing on a GPU, but may
  result in reduced performance.

## FFT size and performance

For performance reasons, when doing FFTs one usually wants the size of the input along each
dimension to be a power of 2 (ideally), or the product of powers of small prime numbers (2,
3, 5, …). The problem is that, with the NUFFT, one does not directly control the FFT size
due to the oversampling factor ``σ``, which may be any real number ``σ > 1``. That is, for
an input of size ``N``, FFTs are performed on an oversampled grid of size ``Ñ ≈ σN``. Note
that ``σN`` is generally not an integer, hence the ``≈``.

The aim of this section is to clarify how ``Ñ`` is actually chosen, so that one can predict
its value for given inputs ``N`` and ``σ``.
This may be better understood in actual code:

```julia
Ñ = nextprod((2, 3, 5), floor(Int, σ * N))
```

Basically, we truncate ``σN`` to an integer, and then we choose ``Ñ`` as the next integer
that can be written as the product of powers of 2, 3 and 5
(see [`nextprod`](https://docs.julialang.org/en/v1/base/math/#Base.nextprod)).
Most often, the result will be greater than or equal to ``σN``.

## Using real non-uniform data

In some applications, the non-uniform data to be transformed is purely real.
In this case, one may pass `Float64` or `Float32` as the first argument.
This may be faster than converting data to complex types, in particular because the
real-to-complex FFTs from FFTW will be used to compute the transforms.
Note that, in this case, the dimensions of the uniform data arrays is not exactly `dims`,
since the size of the first dimension is divided roughly by 2 (taking advantage of Hermitian
symmetry).
For convenience, one can call [`size(::PlanNUFFT)`](@ref) on the constructed plan to know in
advance the dimensions of the uniform data arrays.
"""
struct PlanNUFFT{
        Z <: Number,  # non-uniform data type (can be real or complex)
        N,   # number of dimensions
        Nc,  # number of "components" (simultaneous transforms)
        M,   # kernel half-width
        Backend <: KA.Backend,
        T <: AbstractFloat,  # this is real(Z)
        Kernels <: NTuple{N, AbstractKernelData{<:AbstractKernel, M, T}},
        KernelEvalMode <: EvaluationMode,
        PointsRef <: Ref{<:NTuple{N, AbstractVector{T}}},  # (xs[:], ys[:], ...)
        Data <: AbstractNUFFTData{Z, N, Nc},
        Blocks <: AbstractBlockData,
        IndexMap <: NTuple{N, AbstractVector{Int}},
        Timer <: TimerOutput,
        PointTransform <: Function,
    }
    kernels :: Kernels
    backend :: Backend  # CPU, GPU, ...
    kernel_evalmode :: KernelEvalMode
    σ       :: T       # oversampling factor (≥ 1)
    points_ref  :: PointsRef  # "pointer" to non-uniform points (real values)
    data    :: Data
    blocks  :: Blocks
    fftshift  :: Bool
    index_map :: IndexMap
    timer   :: Timer
    synchronise :: Bool
    point_transform_fold :: PointTransform  # folds points onto the [0, 2π) box (+ optional transforms)
    cpu_use_atomics :: Bool
end

# This represents the type of data in Fourier space.
# This is compatible with the behaviour of `size(::PlanNUFFT)`, which returns the uniform
# array size in Fourier space.
Base.eltype(::PlanNUFFT{Z}) where {Z} = complex(Z)

function Base.show(io::IO, p::PlanNUFFT{Z, N, Nc}) where {Z, N, Nc}
    (; kernels, backend, σ, blocks, fftshift,) = p
    M = Kernels.half_support(first(kernels))
    print(io, "$N-dimensional PlanNUFFT with input type $Z:")
    print(io, "\n  - backend: ", typeof(backend))
    print(io, "\n  - kernel: ", first(kernels))  # should be the same output in all directions
    print(io, "\n  - kernel evaluation mode: ", p.kernel_evalmode)  # should be the same output in all directions
    print(io, "\n  - oversampling factor: σ = ", σ)
    print(io, "\n  - uniform dimensions: ", size(p))
    print(io, "\n  - simultaneous transforms: ", Nc)
    frequency_order = fftshift ? "increasing" : "FFTW"
    print(io, "\n  - frequency order: ", frequency_order, " (fftshift = $fftshift)")
    print(io, "\n  - block size: ")
    if get_block_dims(blocks) === nothing
        print(io, "none (blocking disabled)")
    else
        print(io, get_block_dims(blocks), " (excluding 2M - 1 = $(2M - 1) ghost cells in each direction)")
    end
    print(io, "\n  - internally permuting point order: ", Static.dynamic(get_sort_points(blocks)))
    if backend isa GPU
        method = gpu_method(blocks)
        print(io, "\n  - GPU method: :", method)
        if method === :shared_memory
            @assert hasproperty(blocks, :batch_size)
            print(io, "\n  - batch size for GPU type-1: ", get_batch_size(blocks))
        end
    else
        print(io, "\n  - using atomics in type-1: ", p.cpu_use_atomics)
    end
    nothing
end

get_timer_nowarn(p::PlanNUFFT) = getfield(p, :timer)

# Show warning if timer is retrieved in cases where timings may be incorrect.
function get_timer(p::PlanNUFFT)
    (; backend, synchronise,) = p
    if backend isa GPU && !synchronise
        @warn "synchronisation is disabled on GPU: timings will be incorrect"
    end
    get_timer_nowarn(p)
end

function Base.propertynames(p::PlanNUFFT, private::Bool = false)
    (fieldnames(typeof(p))..., :points)
end

@inline function Base.getproperty(p::PlanNUFFT, name::Symbol)
    if name === :timer
        get_timer(p)
    elseif name === :points
        get_points(p)  # = p.points_ref[]
    else
        getfield(p, name)
    end
end

"""
    size(p::PlanNUFFT) -> (N₁, N₂, ...)

Return the dimensions of arrays containing uniform values.

This corresponds to the number of Fourier modes in each direction (in the non-oversampled grid).
"""
Base.size(p::PlanNUFFT) = map(length, p.data.ks)

Base.ndims(::PlanNUFFT{Z, N}) where {Z, N} = N

"""
    ntransforms(p::PlanNUFFT) -> Int

Return the number of datasets which are simultaneously transformed by a plan.
"""
ntransforms(::PlanNUFFT{Z, N, Nc}) where {Z, N, Nc} = Nc

function get_block_dims(Ñs::Dims, bsize::Int)
    d = length(Ñs)
    bdims = @. false * Ñs + 1  # size along each direction (initially 1 in each direction)
    bprod = 1                  # product of sizes
    i = 1                      # current dimension
    while bprod < bsize
        # Multiply block size by 2 in the current dimension.
        bdims = Base.setindex(bdims, bdims[i] << 1, i)
        bprod <<= 1
        i = ifelse(i == d, 1, i + 1)
    end
    bdims
end

get_block_dims(::Dims{N}, bdims::NTuple{N}) where {N} = bdims

maybe_synchronise(backend::KA.Backend, synchronise::Bool) = synchronise && KA.synchronize(backend)  # this doesn't do anything on the CPU
maybe_synchronise(p::PlanNUFFT) = maybe_synchronise(p.backend, p.synchronise)

# Generate a function that will be used to transform a point and bring it to the [0, 2π) box.
# Note that point_transform is usually identity (so basically free) and is only used by the
# AbstractNFFTs interface to switch between NUFFT conventions.
function generate_point_transform_fold_function(point_transform::F, backend) where {F <: Function}
    @inline function (x)
        y = @inline point_transform(x)  # apply optional transform (usually transform === identity, so this is free)
        @inline to_unit_cell(backend, y)  # fold onto [0, 2π) box
    end
end

# This constructor is generally not called directly.
function _PlanNUFFT(
        ::Type{Z}, kernel::AbstractKernel, h::HalfSupport, σ_wanted, Ns::Dims{D},
        num_transforms::Val;
        timer = TimerOutput(),
        fftw_flags = FFTW.MEASURE,
        fftshift = false,
        sort_points::StaticBool = False(),
        backend::KA.Backend = CPU(),
        kernel_evalmode::EvaluationMode = default_kernel_evalmode(backend),
        block_size::Union{Integer, Dims{D}, Nothing} = default_block_size(Ns, backend),
        synchronise::Bool = false,
        gpu_method::Symbol = :global_memory,
        gpu_batch_size::Val = Val(default_gpu_batch_size(backend)),  # currently only used in shared-memory GPU spreading
        point_transform::F = identity,
        use_atomics::Bool = false,
    ) where {Z <: Number, D, F <: Function}
    ks = init_wavenumbers(Z, Ns)
    # Determine dimensions of oversampled grid.
    Ñs = ntuple(Val(D)) do d
        # We try to make sure that each dimension is a product of powers of small primes,
        # which is good for FFT performance. Moreover, for real-data transforms (rfft),
        # "it is generally beneficial for the last dimension of an r2c/c2r transform
        # to be even" (from the FFTW docs). In our case the "last" dimension is actually the
        # first. This is true for FFTW; not sure about other libraries (including GPU ones).
        if Z <: Real && d == 1
            Ñ = 2 * nextprod((2, 3, 5), floor(Int, σ_wanted * ((Ns[d] + 1) ÷ 2)))  # make sure it's even
        else
            Ñ = nextprod((2, 3, 5), floor(Int, σ_wanted * Ns[d]))
        end
        check_nufft_size(Ñ, h)
        Ñ
    end
    T = real(Z)
    σ::T = maximum(Ñs ./ Ns)  # actual oversampling factor
    kernel_data = map(Ns, Ñs) do N, Ñ
        @inline
        L = T(2π)  # assume 2π period
        Δx̃ = L / Ñ
        Kernels.optimal_kernel(kernel, h, Δx̃, Ñ / N; backend)
    end
    # Precompute Fourier coefficients of the kernels.
    # After doing this, one can call `fourier_coefficients` to get the precomputed
    # coefficients.
    if fftshift
        # Order of Fourier coefficients must match order of output wavenumbers.
        foreach((kdata, kx) -> init_fourier_coefficients!(kdata, AbstractFFTs.fftshift(kx)), kernel_data, ks)
    else
        foreach(init_fourier_coefficients!, kernel_data, ks)
    end
    points = ntuple(_ -> KA.allocate(backend, T, 0), Val(D))  # empty vector of points
    points_ref = Ref(points)
    if block_size === nothing
        blocks = NullBlockData()  # disable blocking (→ can't use multithreading when spreading)
        backend isa CPU && FFTW.set_num_threads(1)   # also disable FFTW threading (avoids allocations)
    else
        block_dims = get_block_dims(Ñs, block_size)
        if backend isa GPU
            blocks = BlockDataGPU(Z, backend, block_dims, Ñs, h, sort_points; method = gpu_method, batch_size = gpu_batch_size,)
        else
            blocks = BlockDataCPU(Z, block_dims, Ñs, h, sort_points)
            FFTW.set_num_threads(Threads.nthreads())
        end
    end
    plan_kwargs = backend isa CPU ? (flags = fftw_flags,) : (;)
    nufft_data = init_plan_data(Z, backend, Ñs, ks, num_transforms; plan_kwargs)
    ûs = first(output_field(nufft_data)) :: AbstractArray{<:Complex}
    index_map = map(ks, axes(ûs)) do k, inds
        indmap = KA.allocate(backend, eltype(inds), length(k))
        non_oversampled_indices!(indmap, k, inds; fftshift)
    end
    point_transform_fold = generate_point_transform_fold_function(point_transform, backend)
    PlanNUFFT(
        kernel_data, backend, kernel_evalmode, σ, points_ref, nufft_data, blocks,
        fftshift, index_map, timer, synchronise, point_transform_fold, use_atomics,
    )
end

@inline get_points(p::PlanNUFFT) = p.points_ref[]

function check_nufft_size(Ñ, ::HalfSupport{M}) where M
    if Ñ < 2 * M
        throw(ArgumentError(
            lazy"""
            data size is too small: σN = $Ñ < $(2M) = 2M. Try either:
                1. increasing the number of data points N 
                2. increasing the oversampling factor σ
                3. decreasing the kernel half-support M"""
        ))
    end
    nothing
end

init_wavenumbers(::Type{T}, Ns::Dims) where {T <: AbstractFloat} = ntuple(Val(length(Ns))) do i
    N = Ns[i]
    # This assumes L = 2π:
    i == 1 ? AbstractFFTs.rfftfreq(N, T(N)) : AbstractFFTs.fftfreq(N, T(N))
end

init_wavenumbers(::Type{Complex{T}}, Ns::Dims) where {T <: AbstractFloat} = map(Ns) do N
    AbstractFFTs.fftfreq(N, T(N))  # this assumes L = 2π
end

function PlanNUFFT(
        ::Type{Z}, Ns::Dims, h::HalfSupport;
        ntransforms = Val(1),
        backend = CPU(),
        kernel::AbstractKernel = default_kernel(backend),
        σ::Real = real(Z)(2), kws...,
    ) where {Z <: Number}
    R = real(Z)
    _PlanNUFFT(Z, kernel, h, R(σ), Ns, to_static(ntransforms); backend, kws...)
end

@inline to_static(ntrans::Val) = ntrans
@inline to_static(ntrans::Int) = Val(ntrans)

# This constructor relies on constant propagation to make the output fully inferred.
Base.@constprop :aggressive function PlanNUFFT(::Type{Z}, Ns::Dims; m = 4, kws...) where {Z <: Number}
    h = to_halfsupport(m)
    PlanNUFFT(Z, Ns, h; kws...)
end

@inline to_halfsupport(m::Integer) = HalfSupport(m)
@inline to_halfsupport(m::HalfSupport) = m

# 1D case
function PlanNUFFT(::Type{Z}, N::Integer, args...; kws...) where {Z <: Number}
    PlanNUFFT(Z, (N,), args...; kws...)
end

# Alternative constructor: use ComplexF64 data by default.
function PlanNUFFT(N::Union{Integer, Dims}, args...; kws...)
    PlanNUFFT(ComplexF64, N, args...; kws...)
end
