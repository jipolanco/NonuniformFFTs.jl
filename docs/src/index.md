# NonuniformFFTs.jl

Yet another package for computing multidimensional [non-uniform fast Fourier transforms (NUFFTs)](https://en.wikipedia.org/wiki/NUFFT) in Julia.

## Installation

NonuniformFFTs.jl can be simply installed from the Julia REPL with:

```julia
julia> ] add NonuniformFFTs
```

## Conventions

### Transform definitions

This package evaluates type-1 (non-uniform to uniform) and type-2 (uniform to
non-uniform) non-uniform fast Fourier transforms (NUFFTs).
These are sometimes also called the *adjoint* and *direct* NUFFTs, respectively.

In one dimension, the type-1 NUFFT computed by this package is defined as follows:

```math
û(k) = ∑_{j = 1}^{M} v_j \, e^{-i k x_j}
\quad \text{ for } \quad
k = -\frac{N}{2}, …, \frac{N}{2} - 1
```

where the ``x_j ∈ [0, 2π)`` are the non-uniform points and the ``v_j`` are the
input values at those points,
and ``k`` are the associated Fourier wavenumbers (or frequencies).
Here ``M`` is the number of non-uniform points, and ``N`` is the number of Fourier modes that are kept
(taken to be even here, but can also be odd).

Similarly, the type-2 NUFFT is defined as:

```math
v_j = ∑_{k = -N/2}^{N/2 + 1} û(k) \, e^{+i k x_j}
```

for ``x_j ∈ [0, 2π)``.
The type-2 transform can be interpreted as an interpolation of a Fourier series onto a given location.

If the points are uniformly distributed in ``[0, 2π)``, i.e. ``x_j = 2π (j - 1) / M``, then these
definitions exactly correspond to the [forward and backward DFTs](http://fftw.org/fftw3_doc/The-1d-Discrete-Fourier-Transform-_0028DFT_0029.html) computed by FFTW.
Note in particular that the type-1 transform is not normalised.
In applications, one usually wants to normalise the obtained Fourier
coefficients by the size ``N`` of the transform (see examples below).

### Ordering of data in frequency space

This package follows the FFTW convention of storing frequency-space data
starting from the non-negative frequencies ``(k = 0, 1, …, N/2 - 1)``, followed
by the negative frequencies ``(k = -N/2, ..., -2, -1)``.
Note that this package also allows the non-uniform data (``v_j`` values) to be purely real,
in which case [real-to-complex FFTs](http://fftw.org/fftw3_doc/The-1d-Real_002ddata-DFT.html) are
performed and only the non-negative wavenumbers are kept (in one dimension).

One can use the
[`fftfreq`](https://juliamath.github.io/AbstractFFTs.jl/stable/api/#AbstractFFTs.fftfreq)
function from the AbstractFFTs package to conveniently obtain the Fourier
frequencies in the right order.
For real data transforms, [`rfftfreq`](https://juliamath.github.io/AbstractFFTs.jl/stable/api/#AbstractFFTs.rfftfreq) should be used instead.

Alternatively, for complex non-uniform data, one can use [`fftshift`](https://juliamath.github.io/AbstractFFTs.jl/stable/api/#AbstractFFTs.fftshift) and
[`ifftshift`](https://juliamath.github.io/AbstractFFTs.jl/stable/api/#AbstractFFTs.ifftshift)
from the same package to switch between this convention and the more
"natural" convention of storing frequencies in increasing order
``(k = -N/2, …, N/2 - 1)``.

## Basic usage

### Type-1 (or *adjoint*) NUFFT in one dimension

```julia
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
plan_nufft = PlanNUFFT(T, N; m = HalfSupport(8))  # larger support increases accuracy

# Set non-uniform points
set_points!(plan_nufft, xp)

# Perform type-1 NUFFT on preallocated output
ûs = Array{Complex{T}}(undef, size(plan_nufft))
exec_type1!(ûs, plan_nufft, vp)
@. ûs = ûs / N  # normalise transform
```

### Type-2 (or *direct*) NUFFT in one dimension

```julia
using NonuniformFFTs

N = 256   # number of Fourier modes
Np = 100  # number of non-uniform points

# Generate some uniform random data
T = Float64                        # non-uniform data is real (can also be complex)
xp = rand(T, Np) .* 2π             # non-uniform points in [0, 2π]
ûs = randn(Complex{T}, N ÷ 2 + 1)  # random values at points (we need to store roughly half the Fourier modes for complex-to-real transform)

# Create plan for data of type T
plan_nufft = PlanNUFFT(T, N; m = HalfSupport(8))

# Set non-uniform points
set_points!(plan_nufft, xp)

# Perform type-2 NUFFT on preallocated output
vp = Array{T}(undef, Np)
exec_type2!(vp, plan_nufft, ûs)
```

### Multidimensional transforms

```julia
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
plan_nufft = PlanNUFFT(T, Ns; m = HalfSupport(8))

# Set non-uniform points
set_points!(plan_nufft, xp)

# Perform type-1 NUFFT on preallocated output
ûs = Array{Complex{T}}(undef, size(plan_nufft))
exec_type1!(ûs, plan_nufft, vp)
ûs ./= prod(Ns)  # normalise transform

# Perform type-2 NUFFT on preallocated output
exec_type2!(vp, plan_nufft, ûs)
```

### Multiple transforms on the same non-uniform points

```julia
using NonuniformFFTs

N = 256   # number of Fourier modes
Np = 100  # number of non-uniform points
ntrans = Val(3)  # number of simultaneous transforms

# Generate some non-uniform random data
T = Float64             # non-uniform data is real (can also be complex)
xp = rand(T, Np) .* 2π  # non-uniform points in [0, 2π]
vp = ntuple(_ -> randn(T, Np), ntrans)  # random values at points (one vector per transformed quantity)

# Create plan for data of type T
plan_nufft = PlanNUFFT(T, N; ntransforms = ntrans)

# Set non-uniform points
set_points!(plan_nufft, xp)

# Perform type-1 NUFFT on preallocated output (one array per transformed quantity)
ûs = ntuple(_ -> Array{Complex{T}}(undef, size(plan_nufft)), ntrans)
exec_type1!(ûs, plan_nufft, vp)
@. ûs = ûs / N  # normalise transform

# Perform type-2 NUFFT on preallocated output (one vector per transformed quantity)
vp_interp = map(similar, vp)
exec_type2!(vp, plan_nufft, ûs)
```

More details on optional parameters and on tuning accuracy is coming soon.

## Differences with other packages

This package roughly follows the same notation and conventions of the [FINUFFT library](https://finufft.readthedocs.io/en/latest/)
and its [Julia interface](https://github.com/ludvigak/FINUFFT.jl), with a few differences detailed below.

### Conventions used by this package

We try to preserve as much as possible the conventions used in FFTW3.
In particular, this means that:

- The FFT outputs are ordered starting from mode $k = 0$ to $k = N/2 - 1$ (for even $N$) and then from $-N/2$ to $-1$.
  Wavenumbers can be obtained in this order by calling `AbstractFFTs.fftfreq(N, N)`.
  Use `AbstractFFTs.fftshift` to get Fourier modes in increasing order $-N/2, …, -1, 0, 1, …, N/2 - 1$.
  In FINUFFT, one should set [`modeord = 1`](https://finufft.readthedocs.io/en/latest/opts.html#data-handling-options) to get this order.

- The type-1 NUFFT (non-uniform to uniform) is defined with a minus sign in the exponential.
  This is the same convention as the [forward DFT in FFTW3](http://fftw.org/fftw3_doc/The-1d-Discrete-Fourier-Transform-_0028DFT_0029.html).
  In particular, this means that performing a type-1 NUFFT on uniform points gives the same output than performing a FFT using FFTW3.
  In FINUFFT, this corresponds to setting [`iflag = -1`](https://ludvigak.github.io/FINUFFT.jl/latest/#FINUFFT.finufft_makeplan-Tuple{Integer,%20Union{Integer,%20Array{Int64}},%20Integer,%20Integer,%20Real}) in type-1 transforms.
  Conversely, type-2 NUFFTs (uniform to non-uniform) are defined with a plus sign, equivalently to the backward DFT in FFTW3.

### Differences with [NFFT.jl](https://github.com/JuliaMath/NFFT.jl)

- This package allows changing the non-uniform points associated to a NUFFT plan.
  In other words, once a plan already exists, computing a NUFFT for a different set of points is efficient and doesn't need to allocate a new plan.

- This package allows NUFFTs of purely real non-uniform data.

- Different convention is used: non-uniform points are expected to be in $[0, 2π]$.

- This package allows performing transforms of multiple quantities at the same non-uniform values at once.

### Differences with FINUFFT / FINUFFT.jl

- This package is written in "pure" Julia (besides the FFTs themselves which rely on the FFTW3 library, via their Julia interface).

- This package allows NUFFTs of purely real non-uniform data.
  Moreover, transforms can be performed in for an arbitrary number of dimensions.

- A different smoothing kernel function is used (backwards Kaiser–Bessel kernel by default).

- It is possible to use the same plan for type-1 and type-2 transforms, reducing memory requirements in cases where one wants to perform both.
