# Changelog

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

### Added

- Add alternative implementation of GPU transforms based on shared-memory arrays.
  This is disabled by default, and can be enabled by passing `gpu_method = :shared_memory` when creating a plan (default is `:global_memory`).

- Add possibility to switch between fast approximation of kernel functions
  (previously the default and only choice) and direct evaluation (previously not implemented).
  These correspond to the new `kernel_evalmode` plan creation option.
  Possible values are `FastApproximation()` and `Direct()`.
  The default depends on the actual backend.
  Currently, `FastApproximation()` is used on CPUs and `Direct()` on GPUs,
  where it is sometimes faster.

- The `AbstractNFFTs.plan_nfft` function is now implemented for full compatibility with the AbstractNFFTs.jl interface.

### Changed

- **BREAKING**: Change default precision of transforms.
  By default, transforms on `Float64` or `ComplexF64` now have a relative precision of the order of $10^{-7}$.
  This corresponds to setting `m = HalfSupport(4)` and oversampling factor `σ = 2.0`.
  Previously, the default was `m = HalfSupport(8)` and `σ = 2.0`, corresponding
  to a relative precision of the order of $10^{-14}$.

- **BREAKING**: The `PlanNUFFT` constructor can no longer be used to create
  plans compatible with AbstractNFFTs.jl / NFFT.jl.
  Instead, a separate (and unexported) `NonuniformFFTs.NFFTPlan` type is now
  defined which may be used for this purpose.
  Alternatively, one can now use the `AbstractNFFTs.plan_nfft` function.

- On GPUs, we now default to direct evaluation of kernel functions (e.g.
  Kaiser-Bessel) instead of polynomial approximations, as this seems to be
  faster and uses far fewer GPU registers.

- On CUDA and AMDGPU, the default kernel is now `KaiserBesselKernel` instead of `BackwardsKaiserBesselKernel`.
  The direct evaluation of the KB kernel (based on Bessel functions) seems to be a bit faster than backwards KB, both on CUDA and AMDGPU.
  Accuracy doesn't change much since both kernels have similar precisions.

## [v0.5.6](https://github.com/jipolanco/NonuniformFFTs.jl/releases/tag/v0.5.6) - 2024-10-17

### Changed

- Minor optimisations and refactoring to GPU kernels.
  Spreading and interpolation operations are slightly faster than before.

## [v0.5.5](https://github.com/jipolanco/NonuniformFFTs.jl/releases/tag/v0.5.5) - 2024-10-04

### Fixed

- Transforms now work on AMD GPUs ([#33](https://github.com/jipolanco/NonuniformFFTs.jl/pull/33)).
  This only required minor modifications to some KA kernels.

## [v0.5.4](https://github.com/jipolanco/NonuniformFFTs.jl/releases/tag/v0.5.4) - 2024-09-25

### Changed

- Removed explicit GPU synchronisation barriers (using `KA.synchronize`) by default.
  This can now be re-enabled by passing `synchronise = true` as a plan argument.
  Enabling synchronisation is useful for getting accurate timings (in `p.timer`) but
  may result in decreased performance.

## [v0.5.3](https://github.com/jipolanco/NonuniformFFTs.jl/releases/tag/v0.5.3) - 2024-09-24

### Changed

- Faster spatial sorting of non-uniform points (CPU and GPU).

- Tune GPU parameters: kernel workgroupsize; block size for spatial sorting.

- Plans: `block_size` argument can now be a tuple (block size along each separate dimension).

## [v0.5.2](https://github.com/jipolanco/NonuniformFFTs.jl/releases/tag/v0.5.2) - 2024-09-23

### Changed

- Avoid recompilation of GPU kernels when number of non-uniform points changes.

## [v0.5.1](https://github.com/jipolanco/NonuniformFFTs.jl/releases/tag/v0.5.1) - 2024-09-20

### Fixed

- Fix transforms of real non-uniform data on CUDA.jl.

## [v0.5.0](https://github.com/jipolanco/NonuniformFFTs.jl/releases/tag/v0.5.0) - 2024-09-20

### Added

- Add preliminary GPU support.

## [v0.4.1](https://github.com/jipolanco/NonuniformFFTs.jl/releases/tag/v0.4.1) - 2024-09-14

### Fixed

- AbstractNFFTs interface: fix 1D transforms.

## [v0.4.0](https://github.com/jipolanco/NonuniformFFTs.jl/releases/tag/v0.4.0) - 2024-09-13

### Added

- Implement [AbstractNFFTs](https://juliamath.github.io/NFFT.jl/stable/abstract/)
  interface for easier comparison with other NUFFT packages.
