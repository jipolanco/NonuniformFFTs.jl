# API

```@meta
CurrentModule = NonuniformFFTs
```

## Creating plans

```@docs
PlanNUFFT
NFFTPlan
```

## Setting non-uniform points

```@docs
set_points!
```

## Executing plans

```@docs
exec_type1!
exec_type2!
```

## Using callbacks

```@docs
NUFFTCallbacks
```

## Other functions

```@docs
size(::PlanNUFFT)
```

## Available kernel functions

```@docs
KaiserBesselKernel
BackwardsKaiserBesselKernel
GaussianKernel
BSplineKernel
```

## Kernel evaluation methods

```@docs
Direct
FastApproximation
```

## Internals

```@docs
spread_from_point!
ntransforms
```

### Kernel functions

```@docs
Kernels.AbstractKernel
Kernels.AbstractKernelData
Kernels.KaiserBesselKernelData
Kernels.GaussianKernelData
Kernels.BSplineKernelData
Kernels.half_support
Kernels.order
Kernels.init_fourier_coefficients!
Kernels.fourier_coefficients
```

## Index

```@index
Pages = ["API.md"]
```
