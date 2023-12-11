# NonuniformFFTs.jl

[![Build Status](https://github.com/jipolanco/NonuniformFFTs.jl/workflows/CI/badge.svg)](https://github.com/jipolanco/NonuniformFFTs.jl/actions)
[![Coverage](https://codecov.io/gh/jipolanco/NonuniformFFTs.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/jipolanco/NonuniformFFTs.jl)

Yet another package for computing multidimensional [non-uniform fast Fourier transforms (NUFFTs)](https://en.wikipedia.org/wiki/NUFFT) in Julia.

## Differences with other packages

This package roughly follows the same notation and conventions of the [FINUFFT library](https://finufft.readthedocs.io/en/latest/)
and its [Julia interface](https://github.com/ludvigak/FINUFFT.jl), with a few differences detailed below.

For now, parallelism is not supported by this package, but this will come in the near future.
On a single thread, performance is comparable (and often better) than other libraries, including those mentioned below.

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

- Setting non-uniform points and executing plans is allocation-free.

- Different convention is used: non-uniform points are expected to be in $[0, 2π]$.

### Differences with FINUFFT / FINUFFT.jl

- This package is written in "pure" Julia (besides the FFTs themselves which rely on the FFTW3 library, via their Julia interface).

- This package allows NUFFTs of purely real non-uniform data.
  Moreover, transforms can be performed in for an arbitrary number of dimensions.

- A different smoothing kernel function is used (backwards Kaiser–Bessel kernel by default).

- Unlike the FINUFFT.jl interface, this package guarantees zero allocations when setting non-uniform points and executing plans.
