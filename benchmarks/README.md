# Benchmarks

The benchmark consists in type-1 and type-2 NUFFTs on a uniform 3D grid of
fixed dimensions $M^3 = 256^3$ (excluding oversampling). We vary the number of
non-uniform points $N$, so that the point density $ρ = N / M^3$ takes values
between $10^{-4}$ (very few points) and $10^1$ (very dense).
Points are randomly located in $[0, 2π)^3$.
Moreover, the relative tolerance is fixed to $10^{-6}$.
In NonuniformFFTs.jl, this can be achieved with the parameters `σ = 1.5`
(oversampling factor) and $m = HalfSupport(4)$ (see Accuracy page in the docs).

The tests were run on a cluster with an AMD EPYC 7302 CPU (32 threads) and an
NVIDIA A100 GPU.

The benchmarks compare NonuniformFFTs.jl v0.6.7 (26/11/2024) and FINUFFT v2.3.1.

Each reported time includes (1) the time spent processing non-uniform points
(`set_points!` / `(cu)finufft_setpts!`) and (2) the time spent on the actual transform (`exec_type{1,2}!` / `(cu)finufft_exec!`).

## FINUFFT set-up

We used FINUFFT via its Julia wrapper [FINUFFT.jl](https://github.com/ludvigak/FINUFFT.jl) v3.3.0. For
performance reasons, the (Cu)FINUFFT libraries were compiled locally and the
FINUFFT.jl sources were modified accordingly as described
[here](https://github.com/ludvigak/FINUFFT.jl?tab=readme-ov-file#advanced-installation-and-locally-compiling-binaries).
FINUFFT was compiled with GCC 10.2.0 using CMake with its default flags in `Release` mode, which include `-fPIC -funroll-loops -O3 -march=native`.
Moreover, we set `CMAKE_CUDA_ARCHITECTURES=80` (for an NVIDIA A100) and used the `nvcc` compiler included in CUDA 12.3.

All FINUFFT benchmarks were run with relative tolerance `1e-6`.
Moreover, the following options were used:

- `modeord = 1` (use FFTW ordering, for consistency with NonuniformFFTs)
- `spread_sort = 1` (enable point sorting in CPU plans)
- `spread_kerevalmeth = 1` (use the recommended piecewise polynomial evaluation)
- `fftw = FFTW.ESTIMATE` (CPU plans)

and for GPU plans:

- `gpu_sort = 1` (enable point sorting)
- `gpu_kerevalmeth = 1` (use piecewise polynomial evaluation)
- `gpu_method = 1` (global memory method, non-uniform point driven)

We also tried `gpu_method = 2` (based on shared memory) but found it to be
considerably slower in almost all cases (in three dimensions, at the requested tolerance).
