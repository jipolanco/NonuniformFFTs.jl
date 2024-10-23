# Changelog

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

### Added

- Add alternative implementation of GPU transforms based on shared-memory arrays.
  This is disabled by default, and can be enabled by passing `gpu_method = :shared_memory`
  (default is `:global_memory`).

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
