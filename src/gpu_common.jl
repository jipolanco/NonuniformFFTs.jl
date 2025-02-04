# Total amount of shared memory available (in bytes).
# This might be overridden in package extensions for specific backends.
available_static_shared_memory(backend::KA.Backend) = Int32(48) << 10  # 48 KiB (usual in CUDA)

# Minimum size of a batch (in number of non-uniform points) used in the shared-memory
# implementation of GPU spreading.
const DEFAULT_GPU_BATCH_SIZE = 16

# Return ndrange parameter to be passed to KA kernels defining shared memory implementations.
gpu_shmem_ndrange_from_groupsize(groupsize::Integer, ngroups::Tuple) =
    Base.setindex(ngroups, groupsize * ngroups[1], 1)  # ngroups[1] -> ngroups[1] * groupsize

# Determine block size if using the shared-memory implementation.
# We try to make sure that the total block size (including 2M - 1 ghost cells in each direction)
# is not larger than the available shared memory. In CUDA the limit is usually 48 KiB.
# Note that the result is a compile-time constant (on Julia 1.11.1 at least).
# For this to be true, the available_static_shared_memory function should also return a
# compile-time constant (see CUDA and AMDGPU extensions for details).
@inline function block_dims_gpu_shmem(
        backend, ::Type{Z}, ::Dims{D}, ::HalfSupport{M}, ::Val{Np_min} = Val(DEFAULT_GPU_BATCH_SIZE);
        warn = false,
    ) where {Z <: Number, D, M, Np_min}
    T = real(Z)
    # These are extra shared-memory needs in spreading kernel (see spread_from_points_shmem_kernel!).
    max_shmem = available_static_shared_memory(backend)  # hopefully this should be a compile-time constant!

    # We try to maximise use of the available shared memory. We split the total capacity as:
    #
    #   max_shmem ≥ const_shmem + shmem_for_local_grid + Np * shmem_per_point
    #
    # where const_shmem and shmem_per_point are defined based on variables defined in
    # spread_from_points_shmem_kernel! (spreading/gpu.jl).

    # (1) Constant shared memory requirement (independent of Np or local grid size)
    # See spread_from_points_shmem_kernel! for the meaning of each variable to the right of
    # each value.
    const_shmem = sizeof(Int) * (
        + 2   # buf_sm
        + D   # ishifts_sm
    ) + 128   # extra 128 bytes for safety (CUDA seems to use slightly more shared memory than what we estimate, maybe due to memory alignment?)

    # (2) Shared memory required per point in a batch
    shmem_per_point = sizeof(T) * D * (
        + 2M  # window_vals
    ) + (
        sizeof(Int) * D   # inds_start
    ) + (
        sizeof(Z)         # vp_sm
    )

    # (3) Determine local grid size based on above values
    max_shmem_localgrid = max_shmem - const_shmem - Np_min * shmem_per_point  # maximum shared memory for local grid (single block)
    max_localgrid_length = max_shmem_localgrid ÷ sizeof(Z)  # maximum number of elements in a block
    m = floor(Int, _invpow(max_localgrid_length, Val(D)))  # local grid size in each direction (including ghost cells / padding)
    n = m - (2M - 1)  # exclude ghost cells
    block_dims = ntuple(_ -> n, Val(D))  # = (n, n, ...)

    if warn
        if n ≤ 0
            throw(ArgumentError(
                lazy"""
                GPU shared memory size is too small for the chosen problem:
                - element type: Z = $Z
                - half-support: M = $M
                - number of dimensions: D = $D
                - minimum spreading batch size: Np = $Np_min
                If possible, reduce some of these parameters, or else switch to gpu_method = :global_memory."""
            ))
        elseif n < 4  # the n < 4 limit is completely empirical
            @warn lazy"""
                GPU shared memory size might be too small for the chosen problem.
                Switching to gpu_method = :global_memory will likely be faster.
                Current parameters:
                - element type: Z = $Z
                - half-support: M = $M
                - number of dimensions: D = $D
                - minimum spreading batch size: Np = $Np_min
                This gives blocks of dimensions $block_dims (not including 2M - 1 = $(2M - 1) ghost cells in each direction).
                If possible, reduce some of these parameters, or else switch to gpu_method = :global_memory."""
        end
    end

    # (4) Now determine Np according to the actual remaining shared memory.
    # Note that, if Np was passed as input, we may increase it to maximise memory usage.
    shmem_left = max_shmem - const_shmem - sizeof(Z) * m^D
    Np_actual = shmem_left ÷ shmem_per_point
    @assert Np_actual ≥ Np_min

    # Actual memory requirement
    # shmem_for_local_grid = m^D * sizeof(Z)
    # required_shmem = const_shmem + shmem_for_local_grid + Np_actual * shmem_per_point
    # @show required_shmem max_shmem

    block_dims, Np_actual
end

# Compute x^(1/D)
_invpow(x, ::Val{1}) = x
_invpow(x, ::Val{2}) = sqrt(x)
_invpow(x, ::Val{3}) = cbrt(x)
_invpow(x, ::Val{4}) = sqrt(sqrt(x))

# This is called from the global memory (naive) implementation of spreading and interpolation kernels.
@inline function get_inds_vals_gpu(gs::NTuple{D}, evalmode::EvaluationMode, points::NTuple{D}, Ns::NTuple{D}, j::Integer) where {D}
    ntuple(Val(D)) do n
        @inline
        get_inds_vals_gpu(gs[n], evalmode, points[n], Ns[n], j)
    end
end

@inline function get_inds_vals_gpu(g::AbstractKernelData, evalmode::EvaluationMode, points::AbstractVector, N::Integer, j::Integer)
    x = @inbounds points[j]
    gdata = Kernels.evaluate_kernel(evalmode, g, x)
    vals = gdata.values    # kernel values
    M = Kernels.half_support(g)
    i₀ = gdata.i - M  # active region is (i₀ + 1):(i₀ + 2M) (up to periodic wrapping)
    i₀ = ifelse(i₀ < 0, i₀ + N, i₀)  # make sure i₀ ≥ 0
    i₀ => vals
end
