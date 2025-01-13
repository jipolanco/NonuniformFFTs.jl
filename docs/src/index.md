# NonuniformFFTs.jl

Yet another package for computing multidimensional [non-uniform fast Fourier transforms (NUFFTs)](https://en.wikipedia.org/wiki/NUFFT) in Julia.

Like other [existing packages](#similar-packages), computation of NUFFTs on CPU
are parallelised using threads.
Transforms can also be performed on GPUs.
In principle all kinds of GPU for which
a [KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl)
backend exists are supported.

## Installation

NonuniformFFTs.jl can be simply installed from the Julia REPL with:

```julia
julia> ] add NonuniformFFTs
```

## [Conventions](@id nufft-conventions)

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

For complex non-uniform data, one can use [`fftshift`](https://juliamath.github.io/AbstractFFTs.jl/stable/api/#AbstractFFTs.fftshift) and
[`ifftshift`](https://juliamath.github.io/AbstractFFTs.jl/stable/api/#AbstractFFTs.ifftshift)
from the same package to switch between this convention and the more
"natural" convention of storing frequencies in increasing order
``(k = -N/2, …, N/2 - 1)``.

Alternatively, one can pass `fftshift = true` to the [`PlanNUFFT`](@ref)
constructor to reorder Fourier modes in increasing order of frequencies ("natural" order).

## [Differences with other packages](@id similar-packages)

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

For compatibility with other packages such as [NFFT.jl](https://github.com/JuliaMath/NFFT.jl), these conventions are *not*
applied when the [AbstractNFFTs.jl interface](@ref AbstractNFFTs-interface) is used.
In this specific case, modes are assumed to be ordered in increasing order, and
the opposite sign convention is used for Fourier transforms.

### Differences with NFFT.jl

- This package allows NUFFTs of purely real non-uniform data.

- Different convention is used: non-uniform points are expected to be in $[0, 2π]$.

### Differences with FINUFFT / cuFINUFFT / FINUFFT.jl

- This package is written in "pure" Julia (besides the FFTs themselves which rely on the FFTW3 library, via their Julia interface).

- This package provides a generic and efficient GPU implementation thanks to
  [KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl)
  meaning that many kinds of GPUs are supported, including not only Nvidia GPUs but
  also AMD ones and possibly more.

- This package allows NUFFTs of purely real non-uniform data.
  Moreover, transforms can be performed on arbitrary number of dimensions.

- A different smoothing kernel function is used (backwards Kaiser–Bessel kernel by default on CPUs; Kaiser–Bessel kernel on GPUs).

- It is possible to use the same plan for type-1 and type-2 transforms, reducing memory requirements in cases where one wants to perform both.

## Bibliography

```@bibliography
```
