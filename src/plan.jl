abstract type AbstractNUFFTData{T <: Number, N, Nc} end

struct RealNUFFTData{
        T <: AbstractFloat, N, Nc,
        WaveNumbers <: NTuple{N, AbstractVector{T}},
        PlanFFT_fw <: FFTW.Plan{T},
        PlanFFT_bw <: FFTW.Plan{Complex{T}},
    } <: AbstractNUFFTData{T, N, Nc}
    ks      :: WaveNumbers  # wavenumbers in *non-oversampled* Fourier grid
    us      :: NTuple{Nc, Array{T, N}}  # values in oversampled grid
    ûs      :: NTuple{Nc, Array{Complex{T}, N}}  # Fourier coefficients in oversampled grid
    plan_fw :: PlanFFT_fw
    plan_bw :: PlanFFT_bw
end

struct ComplexNUFFTData{
        T <: AbstractFloat, N, Nc,
        WaveNumbers <: NTuple{N, AbstractVector{T}},
        PlanFFT_fw <: FFTW.Plan{Complex{T}},
        PlanFFT_bw <: FFTW.Plan{Complex{T}},
    } <: AbstractNUFFTData{Complex{T}, N, Nc}
    ks      :: WaveNumbers
    us      :: NTuple{Nc, Array{Complex{T}, N}}
    plan_fw :: PlanFFT_fw  # in-place transform
    plan_bw :: PlanFFT_bw  # inverse in-place transform
end

# Here the "input" means the gridded data in "physical" space, while the "output"
# corresponds to its Fourier coefficients.
output_field(data::RealNUFFTData) = data.ûs
output_field(data::ComplexNUFFTData) = data.us  # output === input

# Case of real data
function init_plan_data(
        ::Type{T}, Ñs::Dims, ks::NTuple, ::Val{Nc}; fftw_flags,
    ) where {T <: AbstractFloat, Nc}
    @assert Nc ≥ 1
    us = ntuple(_ -> Array{T}(undef, Ñs), Val(Nc))
    dims_out = (Ñs[1] ÷ 2 + 1, Base.tail(Ñs)...)
    ûs = ntuple(_ -> Array{Complex{T}}(undef, dims_out), Val(Nc))
    plan_fw = FFTW.plan_rfft(first(us); flags = fftw_flags)
    plan_bw = FFTW.plan_brfft(first(ûs), Ñs[1]; flags = fftw_flags)
    RealNUFFTData(ks, us, ûs, plan_fw, plan_bw)
end

# Case of complex data
function init_plan_data(
        ::Type{Complex{T}}, Ñs::Dims, ks::NTuple, ::Val{Nc}; fftw_flags,
    ) where {T <: AbstractFloat, Nc}
    @assert Nc ≥ 1
    us = ntuple(_ -> Array{Complex{T}}(undef, Ñs), Val(Nc))
    plan_fw = FFTW.plan_fft!(first(us); flags = fftw_flags)
    plan_bw = FFTW.plan_bfft!(first(us); flags = fftw_flags)
    ComplexNUFFTData(ks, us, plan_fw, plan_bw)
end

"""
    PlanNUFFT([T = ComplexF64], dims::Dims; ntransforms = Val(1), kwargs...)

Construct a plan for performing non-uniform FFTs (NUFFTs).

The created plan contains all data needed to perform NUFFTs for non-uniform data of type `T`
(`ComplexF64` by default) and uniform data with dimensions `dims`.

# Optional keyword arguments

- `ntransforms = Val(1)`: the number of simultaneous transforms to perform.
  This is useful if one wants to transform multiple scalar quantities at the same
  non-uniform points.

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

- `fftw_flags = FFTW.MEASURE`: parameters passed to the FFTW planner.

- `timer = TimerOutput()`: allows to specify a `TimerOutput` (from the
  [TimerOutputs.jl](https://github.com/KristofferC/TimerOutputs.jl) package) where timing
  information will be written to.
  By default the plan creates its own timer.
  One can visualise the time spent on different parts of the NUFFT computation using `p.timer`.

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

"""
struct PlanNUFFT{
        T <: Number, N, Nc, M,
        Treal <: AbstractFloat,  # this is real(T)
        Kernels <: NTuple{N, AbstractKernelData{<:AbstractKernel, M, Treal}},
        Points <: StructVector{NTuple{N, Treal}},
        Data <: AbstractNUFFTData{T, N, Nc},
        Blocks <: AbstractBlockData,
        Timer <: TimerOutput,
    }
    kernels :: Kernels
    σ       :: Treal   # oversampling factor (≥ 1)
    points  :: Points  # non-uniform points (real values)
    data    :: Data
    blocks  :: Blocks
    timer   :: Timer
end

"""
    size(p::PlanNUFFT) -> (N₁, N₂, ...)

Return the dimensions of arrays containing uniform values.

This corresponds to the number of Fourier modes in each direction (in the non-oversampled grid).
"""
Base.size(p::PlanNUFFT) = map(length, p.data.ks)

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
        block_size::Union{Integer, Nothing} = default_block_size(),
        sort_points::StaticBool = False(),
    ) where {T <: Number, D}
    ks = init_wavenumbers(T, Ns)
    # Determine dimensions of oversampled grid.
    Ñs = map(Ns) do N
        # We try to make sure that each dimension is a product of powers of small primes,
        # which is good for FFT performance.
        nextprod((2, 3, 5), floor(Int, σ_wanted * N))
    end
    Tr = real(T)
    σ::Tr = maximum(Ñs ./ Ns)  # actual oversampling factor
    kernel_data = map(Ns, Ñs) do N, Ñ
        @inline
        L = Tr(2π)  # assume 2π period
        Δx̃ = L / Ñ
        Kernels.optimal_kernel(kernel, h, Δx̃, Ñ / N)
    end
    # Precompute Fourier coefficients of the kernels.
    # After doing this, one can call `fourier_coefficients` to get the precomputed
    # coefficients.
    foreach(init_fourier_coefficients!, kernel_data, ks)
    points = StructVector(ntuple(_ -> Tr[], Val(D)))
    if block_size === nothing
        blocks = NullBlockData()  # disable blocking (→ can't use multithreading when spreading)
        FFTW.set_num_threads(1)   # also disable FFTW threading (avoids allocations)
    else
        block_dims = get_block_dims(Ñs, block_size)
        blocks = BlockData(T, block_dims, Ñs, h, num_transforms, sort_points)
        FFTW.set_num_threads(Threads.nthreads())
    end
    nufft_data = init_plan_data(T, Ñs, ks, num_transforms; fftw_flags)
    PlanNUFFT(kernel_data, σ, points, nufft_data, blocks, timer)
end

init_wavenumbers(::Type{T}, Ns::Dims) where {T <: AbstractFloat} = ntuple(Val(length(Ns))) do i
    N = Ns[i]
    # This assumes L = 2π:
    i == 1 ? FFTW.rfftfreq(N, T(N)) : FFTW.fftfreq(N, T(N))
end

init_wavenumbers(::Type{Complex{T}}, Ns::Dims) where {T <: AbstractFloat} = map(Ns) do N
    FFTW.fftfreq(N, T(N))  # this assumes L = 2π
end

function PlanNUFFT(
        ::Type{T}, Ns::Dims, h::HalfSupport;
        ntransforms = Val(1),
        kernel::AbstractKernel = BackwardsKaiserBesselKernel(),
        σ::Real = real(T)(2), kws...,
    ) where {T <: Number}
    R = real(T)
    _PlanNUFFT(T, kernel, h, R(σ), Ns, to_static(ntransforms); kws...)
end

@inline to_static(ntrans::Val) = ntrans
@inline to_static(ntrans::Int) = Val(ntrans)

# This constructor relies on constant propagation to make the output fully inferred.
function PlanNUFFT(::Type{T}, Ns::Dims; m = 8, kws...) where {T <: Number}
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
