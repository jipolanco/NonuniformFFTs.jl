# API

```@meta
CurrentModule = NonuniformFFTs
```

## Creating plans

```@docs
PlanNUFFT
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

## Other functions

```@docs
size(::PlanNUFFT)
```

## Available spreading kernels

```@docs
KaiserBesselKernel
BackwardsKaiserBesselKernel
GaussianKernel
BSplineKernel
```

## Index

```@index
Pages = ["API.md"]
```
