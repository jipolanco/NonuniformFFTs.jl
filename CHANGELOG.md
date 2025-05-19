# Changelog

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- Tune performance on AMDGPU.
  We now default to `Direct` evaluation (as with CUDA), which can be much
  faster than `FastApproximation`. The difference is more visible in type-2
  transforms, while type-1 doesn't change that much.
  Besides, direct evaluation of the (non-default) `KaiserBesselKernel` has been optimised.

## [v0.7.3] - 2025-04-28

### Changed

- CPU: avoid `:static` scheduling when using threads.
  This improves composability while it shouldn't noticeably impact performance.
  See the [Julia docs](https://docs.julialang.org/en/v1/base/multi-threading/#Base.Threads.@threads) for more details.

## [v0.7.2] - 2025-04-15

### Changed

- Require KernelAbstractions 0.9.34 and avoid implicit bounds checking in shared-memory kernels.
  This may slightly improve performance.

## [v0.7.1] - 2025-02-18

### Fixed

- Fix GPU kernel compilation failures on AMDGPU.

## [v0.7.0] - 2025-02-17

### Changed

- Plans no longer store a _copy_ of the non-uniform point locations. Instead,
  they now store a "pointer" to the points passed by the user to `set_points!`
  (unless the points were passed in an incompatible format, such as a vector of
  `SVector`s, in which case a copy is made). This should reduce memory
  requirements as well as improve `set_points!` performance by avoiding discontiguous
  memory writes.

  This is slightly breaking because _points should no longer be modified_ by the
  user between calls to `set_points!` and `exec_type*!`. This was previously allowed
  since we made a copy in `set_points!`. Note that this is basically the same
  behaviour of FINUFFT.

## [v0.6.8] - 2025-02-12

### Added

- Minor GPU performance improvements.

## [v0.6.7] - 2024-11-26

### Fixed

- Avoid error when creating high-accuracy GPU plans.
  This affected plans that cannot be treated using the `:shared_memory` method
  (because they require large memory buffers), such as plans with `ComplexF64`
  data associated to a large kernel width (e.g. `HalfSupport(8)`). Such plans
  can still be computed using the `:global_memory` method, but this failed up to now.

## [v0.6.6] - 2024-11-25

### Changed

- Improve parallel performance of `set_points!` with `CPU` backend.

## [v0.6.5] - 2024-11-18

### Fixed

- Fix scalar indexing error on latest AMDGPU.jl (v1.1.1).
  Not sure exactly if it's a recent change in AMDGPU.jl, or maybe in GPUArrays.jl, which caused the error.

## [v0.6.4] - 2024-11-17

### Changed

- Avoid large GPU allocation in type-2 transforms when using the CUDA backend.
  The allocation was due to CUDA.jl creating a copy of the input in complex-to-real FFTs
  (see [CUDA.jl#2249](https://github.com/JuliaGPU/CUDA.jl/issues/2249)).

## [v0.6.2] - 2024-11-04

### Changed

- Improve performance of atomic operations (affecting type-1 transforms) on AMD
  GPUs by using `@atomic :monotonic`.

- Change a few defaults on AMD GPUs to improve performance.
  This is based on experiments with an AMD MI210, where the new defaults should give better performance.
  We now default to fast polynomial approximation of kernel functions and to
  the backwards Kaiser-Bessel kernel (as in the CPU).

## [v0.6.1] - 2024-10-29

### Fixed

- Fix type-2 transforms on the GPU when performing multiple transforms at once
  (`ntransforms > 1`) and when `gpu_method = :shared_memory` (which is not currently the default).

## [v0.6.0] - 2024-10-29

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

## [v0.5.6] - 2024-10-17

### Changed

- Minor optimisations and refactoring to GPU kernels.
  Spreading and interpolation operations are slightly faster than before.

## [v0.5.5] - 2024-10-04

### Fixed

- Transforms now work on AMD GPUs ([#33](https://github.com/jipolanco/NonuniformFFTs.jl/pull/33)).
  This only required minor modifications to some KA kernels.

## [v0.5.4] - 2024-09-25

### Changed

- Removed explicit GPU synchronisation barriers (using `KA.synchronize`) by default.
  This can now be re-enabled by passing `synchronise = true` as a plan argument.
  Enabling synchronisation is useful for getting accurate timings (in `p.timer`) but
  may result in decreased performance.

## [v0.5.3] - 2024-09-24

### Changed

- Faster spatial sorting of non-uniform points (CPU and GPU).

- Tune GPU parameters: kernel workgroupsize; block size for spatial sorting.

- Plans: `block_size` argument can now be a tuple (block size along each separate dimension).

## [v0.5.2] - 2024-09-23

### Changed

- Avoid recompilation of GPU kernels when number of non-uniform points changes.

## [v0.5.1] - 2024-09-20

### Fixed

- Fix transforms of real non-uniform data on CUDA.jl.

## [v0.5.0] - 2024-09-20

### Added

- Add preliminary GPU support.

## [v0.4.1] - 2024-09-14

### Fixed

- AbstractNFFTs interface: fix 1D transforms.

## [v0.4.0] - 2024-09-13

### Added

- Implement [AbstractNFFTs](https://juliamath.github.io/NFFT.jl/stable/abstract/)
  interface for easier comparison with other NUFFT packages.

  [unreleased]: https://github.com/jipolanco/NonuniformFFTs.jl/compare/v0.7.2...HEAD
  [v0.7.2]: https://github.com/jipolanco/NonuniformFFTs.jl/releases/tag/v0.7.2
  [v0.7.1]: https://github.com/jipolanco/NonuniformFFTs.jl/releases/tag/v0.7.1
  [v0.7.0]: https://github.com/jipolanco/NonuniformFFTs.jl/releases/tag/v0.7.0
  [v0.6.8]: https://github.com/jipolanco/NonuniformFFTs.jl/releases/tag/v0.6.8
  [v0.6.7]: https://github.com/jipolanco/NonuniformFFTs.jl/releases/tag/v0.6.7
  [v0.6.6]: https://github.com/jipolanco/NonuniformFFTs.jl/releases/tag/v0.6.6
  [v0.6.5]: https://github.com/jipolanco/NonuniformFFTs.jl/releases/tag/v0.6.5
  [v0.6.4]: https://github.com/jipolanco/NonuniformFFTs.jl/releases/tag/v0.6.4
  [v0.6.2]: https://github.com/jipolanco/NonuniformFFTs.jl/releases/tag/v0.6.2
  [v0.6.1]: https://github.com/jipolanco/NonuniformFFTs.jl/releases/tag/v0.6.1
  [v0.6.0]: https://github.com/jipolanco/NonuniformFFTs.jl/releases/tag/v0.6.0
  [v0.5.6]: https://github.com/jipolanco/NonuniformFFTs.jl/releases/tag/v0.5.6
  [v0.5.5]: https://github.com/jipolanco/NonuniformFFTs.jl/releases/tag/v0.5.5
  [v0.5.4]: https://github.com/jipolanco/NonuniformFFTs.jl/releases/tag/v0.5.4
  [v0.5.3]: https://github.com/jipolanco/NonuniformFFTs.jl/releases/tag/v0.5.3
  [v0.5.2]: https://github.com/jipolanco/NonuniformFFTs.jl/releases/tag/v0.5.2
  [v0.5.1]: https://github.com/jipolanco/NonuniformFFTs.jl/releases/tag/v0.5.1
  [v0.5.0]: https://github.com/jipolanco/NonuniformFFTs.jl/releases/tag/v0.5.0
  [v0.4.1]: https://github.com/jipolanco/NonuniformFFTs.jl/releases/tag/v0.4.1
  [v0.4.0]: https://github.com/jipolanco/NonuniformFFTs.jl/releases/tag/v0.4.0
