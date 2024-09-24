# Changelog

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

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
