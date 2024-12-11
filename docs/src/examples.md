# Examples

## Basic usage

The first two examples illustrate how to perform type-1 and type-2 NUFFTs in one dimension.
In these examples the non-uniform data is real-valued, but it could be easily made
complex by replacing `Float64` with `ComplexF64`.

### Type-1 (or *adjoint*) NUFFT in one dimension

```@example
using NonuniformFFTs
using AbstractFFTs: rfftfreq  # can be used to obtain the associated Fourier wavenumbers

N = 256   # number of Fourier modes
Np = 100  # number of non-uniform points
ks = rfftfreq(N, N)  # Fourier wavenumbers

# Generate some non-uniform random data
T = Float64             # non-uniform data is real (can also be complex)
xp = rand(T, Np) .* 2π  # non-uniform points in [0, 2π]
vp = randn(T, Np)       # random values at points

# Create plan for data of type T
plan_nufft = PlanNUFFT(T, N; m = HalfSupport(4))  # larger support increases accuracy

# Set non-uniform points
set_points!(plan_nufft, xp)

# Perform type-1 NUFFT on preallocated output
ûs = Array{Complex{T}}(undef, size(plan_nufft))
exec_type1!(ûs, plan_nufft, vp)
nothing  # hide
```

### Type-2 (or *direct*) NUFFT in one dimension

```@example
using NonuniformFFTs

N = 256   # number of Fourier modes
Np = 100  # number of non-uniform points

# Generate some uniform random data
T = Float64                        # non-uniform data is real (can also be complex)
xp = rand(T, Np) .* 2π             # non-uniform points in [0, 2π]
ûs = randn(Complex{T}, N ÷ 2 + 1)  # random values at points (we need to store roughly half the Fourier modes for complex-to-real transform)

# Create plan for data of type T
plan_nufft = PlanNUFFT(T, N; m = HalfSupport(4))

# Set non-uniform points
set_points!(plan_nufft, xp)

# Perform type-2 NUFFT on preallocated output
vp = Array{T}(undef, Np)
exec_type2!(vp, plan_nufft, ûs)
nothing  # hide
```

## Multidimensional transforms

To perform a ``d``-dimensional transform, one simply needs to pass a tuple of
dimensions `(Nx, Ny, …)` to `PlanNUFFT`.
Moreover, the vector of ``d``-dimensional positions can either be specified as a vector of tuples `(xᵢ, yᵢ, …)`
or, similarly, as a vector of `SVector`s:

```@example
using NonuniformFFTs
using StaticArrays: SVector  # for convenience

Ns = (256, 256)  # number of Fourier modes in each direction
Np = 1000        # number of non-uniform points

# Generate some non-uniform random data
T = Float64                                   # non-uniform data is real (can also be complex)
d = length(Ns)                                # number of dimensions (d = 2 here)
xp = [2π * rand(SVector{d, T}) for _ ∈ 1:Np]  # non-uniform points in [0, 2π]ᵈ
vp = randn(T, Np)                             # random values at points

# Create plan for data of type T
plan_nufft = PlanNUFFT(T, Ns; m = HalfSupport(4))

# Set non-uniform points
set_points!(plan_nufft, xp)

# Perform type-1 NUFFT on preallocated output
ûs = Array{Complex{T}}(undef, size(plan_nufft))
exec_type1!(ûs, plan_nufft, vp)

# Perform type-2 NUFFT on preallocated output
exec_type2!(vp, plan_nufft, ûs)
nothing  # hide
```

## Multiple transforms on the same non-uniform points

One may want to perform multiple transforms with the same non-uniform points.
For example, this can be useful for dealing with vector quantities (as opposed to scalar ones).
To do this, one should pass `ntransforms = Val(Nt)` where `Nt` is the number of transforms to perform.
Moreover, the input and output data should be tuples of arrays of length `Nt`, as shown below.

```@example
using NonuniformFFTs

N = 256   # number of Fourier modes
Np = 100  # number of non-uniform points
ntrans = Val(3)  # number of simultaneous transforms

# Generate some non-uniform random data
T = Float64             # non-uniform data is real (can also be complex)
xp = rand(T, Np) .* 2π  # non-uniform points in [0, 2π]
vp = ntuple(_ -> randn(T, Np), ntrans)  # random values at points (this is a tuple of 3 arrays)

# Create plan for data of type T
plan_nufft = PlanNUFFT(T, N; ntransforms = ntrans)

# Set non-uniform points
set_points!(plan_nufft, xp)

# Perform type-1 NUFFT on preallocated output (one array per transformed quantity)
ûs = ntuple(_ -> Array{Complex{T}}(undef, size(plan_nufft)), ntrans)  # this is a tuple of 3 arrays
exec_type1!(ûs, plan_nufft, vp)

# Perform type-2 NUFFT on preallocated output (one vector per transformed quantity)
vp_interp = map(similar, vp)  # this is a tuple of 3 arrays
exec_type2!(vp, plan_nufft, ûs)
nothing  # hide
```

## GPU usage

Below is a GPU version of the multidimensional transform example above.
The only differences are:

- we import [CUDA.jl](https://github.com/JuliaGPU/CUDA.jl)
- we import [Adapt.jl](https://github.com/JuliaGPU/Adapt.jl) (optional but convenient)
- we pass `backend = CUDABackend()` to `PlanNUFFT` (`CUDABackend` is a [KernelAbstractions backend](https://juliagpu.github.io/KernelAbstractions.jl/stable/#Supported-backends) and is exported by CUDA.jl).
  The default is `backend = CPU()`.
- we copy input arrays to the GPU before calling any NUFFT-related functions ([`set_points!`](@ref), [`exec_type1!`](@ref), [`exec_type2!`](@ref))

The example is for an Nvidia GPU (using [CUDA.jl](https://github.com/JuliaGPU/CUDA.jl)), but should also work with e.g. [AMDGPU.jl](https://github.com/JuliaGPU/AMDGPU.jl)
on an AMD GPU by simply choosing `backend = ROCBackend()`.

```julia
using NonuniformFFTs
using StaticArrays: SVector  # for convenience
using CUDA
using Adapt: adapt  # optional (see below)

backend = CUDABackend()  # other options are CPU() or ROCBackend()

Ns = (256, 256)  # number of Fourier modes in each direction
Np = 1000        # number of non-uniform points

# Generate some non-uniform random data
T = Float64                                          # non-uniform data is real (can also be complex)
d = length(Ns)                                       # number of dimensions (d = 2 here)
xp_cpu = [T(2π) * rand(SVector{d, T}) for _ ∈ 1:Np]  # non-uniform points in [0, 2π]ᵈ
vp_cpu = randn(T, Np)                                # random values at points

# Copy data to the GPU (using Adapt is optional but it makes code more generic).
# Note that all data needs to be on the GPU before setting points or executing transforms.
# We could have also generated the data directly on the GPU.
xp = adapt(backend, xp_cpu)  # returns a CuArray if backend = CUDABackend
vp = adapt(backend, vp_cpu)

# Create plan for data of type T
plan_nufft = PlanNUFFT(T, Ns; m = HalfSupport(4), backend)

# Set non-uniform points
set_points!(plan_nufft, xp)

# Perform type-1 NUFFT on preallocated output
ûs = similar(vp, Complex{T}, size(plan_nufft))  # initialises a GPU array for the output
exec_type1!(ûs, plan_nufft, vp)

# Perform type-2 NUFFT on preallocated output
exec_type2!(vp, plan_nufft, ûs)
```

## [AbstractNFFTs.jl interface](@id AbstractNFFTs-interface)

This package also implements the [AbstractNFFTs.jl](https://juliamath.github.io/NFFT.jl/stable/abstract/)
interface as an alternative API for constructing plans and evaluating transforms.
This can be useful for comparing with similar packages such as [NFFT.jl](https://github.com/JuliaMath/NFFT.jl).

For this, a [`NFFTPlan`](@ref NonuniformFFTs.NFFTPlan) constructor (alternative to [`PlanNUFFT`](@ref)) is provided which
supports most of the parameters supported by [NFFT.jl](https://github.com/JuliaMath/NFFT.jl).
Alternatively, once NonuniformFFTs.jl has been loaded, the [`plan_nfft`](https://juliamath.github.io/NFFT.jl/v0.13.5/api/#AbstractNFFTs.plan_nfft-Union{Tuple{D},%20Tuple{T},%20Tuple{Type{%3C:Array},%20Matrix{T},%20Tuple{Vararg{Int64,%20D}},%20Vararg{Any}}}%20where%20{T,%20D})
function from AbstractNFFTs.jl also generates a `NFFTPlan`.
For compatibility with NFFT.jl, the plan generated via this interface **does not
follow the same [conventions](@ref nufft-conventions)** followed by `PlanNUFFT` plans.

The main differences are:

- points are assumed to be in ``[-1/2, 1/2)`` instead of ``[0, 2π)``;
- the opposite Fourier sign convention is used (e.g. ``e^{-i k x_j}`` becomes ``e^{+2π i k x_j}``);
- uniform data is in increasing order, with frequencies ``k = -N/2, …, -1, 0,
  1, …, N/2-1``, as opposed to preserving the order used by FFTW (which starts at ``k = 0``);
- points locations must be specified as a matrix of dimensions `(d, Np)`.

### Example usage

```@example
using NonuniformFFTs
using AbstractNFFTs: AbstractNFFTs, plan_nfft
using LinearAlgebra: mul!

Ns = (256, 256)  # number of Fourier modes in each direction
Np = 1000        # number of non-uniform points

# Generate some non-uniform random data
T = Float64                      # must be a real data type (Float32, Float64)
d = length(Ns)                   # number of dimensions (d = 2 here)
xp = rand(T, (d, Np)) .- T(0.5)  # non-uniform points in [-1/2, 1/2)ᵈ; must be given as a (d, Np) matrix
vp = randn(Complex{T}, Np)       # random values at points (must be complex)

# Create plan for data of type Complex{T}. Note that we pass the points `xp` as
# a first argument, which calls an AbstractNFFTs-compatible constructor.
p = NonuniformFFTs.NFFTPlan(xp, Ns)
# p = plan_nfft(xp, Ns)  # this is also possible

# Getting the expected dimensions of input and output data.
AbstractNFFTs.size_in(p)   # (256, 256)
AbstractNFFTs.size_out(p)  # (1000,)

# Perform adjoint NFFT, a.k.a. type-1 NUFFT (non-uniform to uniform)
us = adjoint(p) * vp      # allocates output array `us`
mul!(us, adjoint(p), vp)  # uses preallocated output array `us`

# Perform forward NFFT, a.k.a. type-2 NUFFT (uniform to non-uniform)
wp = p * us
mul!(wp, p, us)

# Setting a different set of non-uniform points
AbstractNFFTs.nodes!(p, xp)
nothing  # hide
```

Note: the AbstractNFFTs.jl interface currently only supports complex-valued non-uniform data.
For real-to-complex transforms, the standard NonuniformFFTs.jl API demonstrated [above](#Basic-usage) (based on [`PlanNUFFT`](@ref)) should be used instead.

