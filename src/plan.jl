abstract type AbstractNUFFTData{T <: Number, N, Nc} end

struct RealNUFFTData{
        T <: AbstractFloat, N, Nc,
        WaveNumbers <: NTuple{N, AbstractVector{T}},
        FieldsR <: NTuple{Nc, AbstractArray{T, N}},
        FieldsC <: NTuple{Nc, AbstractArray{Complex{T}, N}},
        PlanFFT_fw <: AbstractFFTs.Plan{T},
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
    us = ntuple(_ -> KA.allocate(backend, T, Ñs), Val(Nc))
    dims_out = (Ñs[1] ÷ 2 + 1, Base.tail(Ñs)...)
    ûs = ntuple(_ -> KA.allocate(backend, Complex{T}, dims_out), Val(Nc))
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
    us = ntuple(_ -> KA.allocate(backend, Complex{T}, Ñs), Val(Nc))
    plan_fw = AbstractFFTs.plan_fft!(first(us); plan_kwargs...)
    plan_bw = AbstractFFTs.plan_bfft!(first(us); plan_kwargs...)
    ComplexNUFFTData(ks, us, plan_fw, plan_bw)
end

"""
    PlanNUFFT([T = ComplexF64], dims::Dims; ntransforms = Val(1), backend = CPU(), kwargs...)

Construct a plan for performing non-uniform FFTs (NUFFTs).

The created plan contains all data needed to perform NUFFTs for non-uniform data of type `T`
(`ComplexF64` by default) and uniform data with dimensions `dims`.

# Optional keyword arguments

- `ntransforms = Val(1)`: the number of simultaneous transforms to perform.
  This is useful if one wants to transform multiple scalar quantities at the same
  non-uniform points.

- `backend::KernelAbstractions.Backend = CPU()`: corresponds to the device type where
  everything will be executed. This could be e.g. `CUDABackend()` if CUDA.jl is loaded.

## NUFFT parameters

- `m = HalfSupport(8)`: the half-support of the convolution kernels. Large values
  increase accuracy at the cost of performance.

- `σ = 2.0`: NUFFT oversampling factor. Typical values are 2.0 (more accurate) and 1.25 (faster),
  but other values such as 1.5 should also work.

- `kernel::AbstractKernel = BackwardsKaiserBesselKernel()`: convolution kernel used for NUFFTs.

## Performance parameters

- `block_size = 4096`: the linear block size (in number of elements) when using block partitioning.
  This can be tuned for maximal performance.
  Using block partitioning is required for running with multiple threads.
  Blocking can be completely disabled by passing `block_size = nothing` (but this is
  generally slower, even when running on a single thread).

- `sort_points = False()`: whether to internally permute the order of the non-uniform points.
  This can be enabled by passing `sort_points = True()`.
  In this case, more time will be spent in [`set_points!`](@ref) and less time on the actual transforms.
  This can improve performance if executing multiple transforms on the same non-uniform points.
  Note that, even when enabled, this does not modify the `points` argument passed to `set_points!`.

## Other parameters

- `fftw_flags = FFTW.MEASURE`: parameters passed to the FFTW planner (only used when `backend = CPU()`).

- `fftshift = false`: determines the order of wavenumbers in uniform space.
  If `false` (default), the same order used by FFTW is used, with positive wavenumbers first
  (`[0, 1, 2, …, N÷2-1]` for even-size transforms) and negative ones afterwards ([-N÷2, …, -1]).
  Otherwise, wavenumbers are expected to be in increasing order ([-N÷2, -kmax, …, -1, 0, 1, …, N÷2-1]),
  which is compatible with the output in NFFT.jl and corresponds to applying the
  `AbstractFFTs.fftshift` function to the data.
  This option also corresponds to the `modeord` parameter in FINUFFT.
  This only affects complex-to-complex transforms.

- `timer = TimerOutput()`: allows to specify a `TimerOutput` (from the
  [TimerOutputs.jl](https://github.com/KristofferC/TimerOutputs.jl) package) where timing
  information will be written to.
  By default the plan creates its own timer.
  One can visualise the time spent on different parts of the NUFFT computation using `p.timer`.

# FFT size and performance

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

# Using real non-uniform data

In some applications, the non-uniform data to be transformed is purely real.
In this case, one may pass `Float64` or `Float32` as the first argument.
This may be faster than converting data to complex types, in particular because the
real-to-complex FFTs from FFTW will be used to compute the transforms.
Note that, in this case, the dimensions of the uniform data arrays is not exactly `dims`,
since the size of the first dimension is divided roughly by 2 (taking advantage of Hermitian
symmetry).
For convenience, one can call [`size(::PlanNUFFT)`](@ref) on the constructed plan to know in
advance the dimensions of the uniform data arrays.

---

    PlanNUFFT(xp::AbstractMatrix{T}, dims::Dims{D}; kwargs...)

Create a [`PlanNUFFT`](@ref) which is compatible with the
[AbstractNFFTs.jl](https://juliamath.github.io/NFFT.jl/stable/abstract/) interface.

This constructor requires passing the non-uniform locations `xp` as the first argument.
These should be given as a matrix of dimensions `(D, Np)`, where `D` is the spatial
dimension and `Np` the number of non-uniform points.

The second argument is simply the size `(N₁, N₂, …)` of the uniform data arrays.

This variant creates a plan which assumes complex-valued non-uniform data.
For real-valued data, the other constructor should be used instead.

# Compatibility with NFFT.jl

Most of the [parameters](https://juliamath.github.io/NFFT.jl/stable/overview/#Parameters)
supported by the NFFT.jl package are also supported by this constructor.
The currently supported parameters are `reltol`, `m`, `σ`, `window`, `blocking`, `sortNodes` and `fftflags`.

Moreover, unlike the first variant, this constructor sets `fftshift = true` by default (but
can be overridden) so that the uniform data ordering is the same as in NFFT.jl.

!!! warning "Type instability"

    Explicitly passing some of these parameters may result in type-unstable code, since the
    exact type of the returned plan cannot be inferred.
    This is because, in NonuniformFFTs.jl, parameters such as the kernel size (`m`) or the
    convolution window (`window`) are included in the plan type (they are compile-time constants).

"""
struct PlanNUFFT{
        T <: Number,  # non-uniform data type (can be real or complex)
        N,   # number of dimensions
        Nc,  # number of "components" (simultaneous transforms)
        M,   # kernel half-width
        Backend <: KA.Backend,
        Treal <: AbstractFloat,  # this is real(T)
        Kernels <: NTuple{N, AbstractKernelData{<:AbstractKernel, M, Treal}},
        Points <: StructVector{NTuple{N, Treal}},
        Data <: AbstractNUFFTData{T, N, Nc},
        Blocks <: AbstractBlockData,
        IndexMap <: NTuple{N, AbstractVector{Int}},
        Timer <: TimerOutput,
    } <: AbstractNFFTPlan{Treal, N, 1}  # the AbstractNFFTPlan only really makes sense when T <: Complex
    kernels :: Kernels
    backend :: Backend  # CPU, GPU, ...
    σ       :: Treal   # oversampling factor (≥ 1)
    points  :: Points  # non-uniform points (real values)
    data    :: Data
    blocks  :: Blocks
    fftshift  :: Bool
    index_map :: IndexMap
    timer   :: Timer
end

function Base.show(io::IO, p::PlanNUFFT{T, N, Nc}) where {T, N, Nc}
    (; kernels, backend, σ, fftshift,) = p
    print(io, "$N-dimensional PlanNUFFT with input type $T:")
    print(io, "\n  - backend: ", typeof(backend))
    print(io, "\n  - kernel: ", first(kernels))  # should be the same output in all directions
    print(io, "\n  - oversampling factor: σ = ", σ)
    print(io, "\n  - uniform dimensions: ", size(p))
    print(io, "\n  - simultaneous transforms: ", Nc)
    frequency_order = fftshift ? "increasing" : "FFTW"
    print(io, "\n  - frequency order: ", frequency_order, " (fftshift = $fftshift)")
    nothing
end

"""
    size(p::PlanNUFFT) -> (N₁, N₂, ...)

Return the dimensions of arrays containing uniform values.

This corresponds to the number of Fourier modes in each direction (in the non-oversampled grid).
"""
Base.size(p::PlanNUFFT) = map(length, p.data.ks)

Base.ndims(::PlanNUFFT{T, N}) where {T, N} = N

"""
    ntransforms(p::PlanNUFFT) -> Int

Return the number of datasets which are simultaneously transformed by a plan.
"""
ntransforms(::PlanNUFFT{T, N, Nc}) where {T, N, Nc} = Nc

default_block_size() = 4096  # in number of linear elements

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

# This constructor is generally not called directly.
function _PlanNUFFT(
        ::Type{T}, kernel::AbstractKernel, h::HalfSupport, σ_wanted, Ns::Dims{D},
        num_transforms::Val;
        timer = TimerOutput(),
        fftw_flags = FFTW.MEASURE,
        fftshift = false,
        block_size::Union{Integer, Nothing} = default_block_size(),
        sort_points::StaticBool = False(),
        backend::KA.Backend = CPU(),
    ) where {T <: Number, D}
    ks = init_wavenumbers(T, Ns)
    # Determine dimensions of oversampled grid.
    Ñs = map(Ns) do N
        # We try to make sure that each dimension is a product of powers of small primes,
        # which is good for FFT performance.
        Ñ = nextprod((2, 3, 5), floor(Int, σ_wanted * N))
        check_nufft_size(Ñ, h)
        Ñ
    end
    Tr = real(T)
    σ::Tr = maximum(Ñs ./ Ns)  # actual oversampling factor
    kernel_data = map(Ns, Ñs) do N, Ñ
        @inline
        L = Tr(2π)  # assume 2π period
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
    points = StructVector(ntuple(_ -> KA.allocate(backend, Tr, 0), Val(D)))  # empty vector of points
    if block_size === nothing || backend isa GPU
        blocks = NullBlockData()  # disable blocking (→ can't use multithreading when spreading)
        backend isa CPU && FFTW.set_num_threads(1)   # also disable FFTW threading (avoids allocations)
    else
        block_dims = get_block_dims(Ñs, block_size)
        blocks = BlockData(T, block_dims, Ñs, h, num_transforms, sort_points)
        backend isa CPU && FFTW.set_num_threads(Threads.nthreads())
    end
    plan_kwargs = backend isa CPU ? (flags = fftw_flags,) : (;)
    nufft_data = init_plan_data(T, backend, Ñs, ks, num_transforms; plan_kwargs)
    ûs = first(output_field(nufft_data)) :: AbstractArray{<:Complex}
    index_map = map(ks, axes(ûs)) do k, inds
        indmap = KA.allocate(backend, eltype(inds), length(k))
        non_oversampled_indices!(indmap, k, inds; fftshift)
    end
    PlanNUFFT(kernel_data, backend, σ, points, nufft_data, blocks, fftshift, index_map, timer)
end

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
        ::Type{T}, Ns::Dims, h::HalfSupport;
        ntransforms = Val(1),
        kernel::AbstractKernel = default_kernel(),
        σ::Real = real(T)(2), kws...,
    ) where {T <: Number}
    R = real(T)
    _PlanNUFFT(T, kernel, h, R(σ), Ns, to_static(ntransforms); kws...)
end

@inline to_static(ntrans::Val) = ntrans
@inline to_static(ntrans::Int) = Val(ntrans)

# This constructor relies on constant propagation to make the output fully inferred.
Base.@constprop :aggressive function PlanNUFFT(::Type{T}, Ns::Dims; m = 8, kws...) where {T <: Number}
    h = to_halfsupport(m)
    PlanNUFFT(T, Ns, h; kws...)
end

@inline to_halfsupport(m::Integer) = HalfSupport(m)
@inline to_halfsupport(m::HalfSupport) = m

# 1D case
function PlanNUFFT(::Type{T}, N::Integer, args...; kws...) where {T <: Number}
    PlanNUFFT(T, (N,), args...; kws...)
end

# Alternative constructor: use ComplexF64 data by default.
function PlanNUFFT(N::Union{Integer, Dims}, args...; kws...)
    PlanNUFFT(ComplexF64, N, args...; kws...)
end
