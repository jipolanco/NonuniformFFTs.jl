# API

```@meta
CurrentModule = NonuniformFFTs
```
## Public API

### Creating plans

```@docs
PlanNUFFT
```

### Setting non-uniform points

```@docs
set_points!
```

### Executing plans

```@docs
exec_type1!
exec_type2!
```

### Available spreading kernels

```@docs
KaiserBesselKernel
BackwardsKaiserBesselKernel
GaussianKernel
BSplineKernel
```
## Internal API
```@autodocs
Modules = [NonuniformFFTs, NonuniformFFTs.Kernels]
Public = false
```

## Index

```@index
Pages = ["api.md"]
```
