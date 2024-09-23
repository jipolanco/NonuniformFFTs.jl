# Changelog

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

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
