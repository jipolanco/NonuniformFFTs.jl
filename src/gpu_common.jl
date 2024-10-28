# Group size used in shared-memory implementation of spreading and interpolation.
# Input variables are not currently used (except for the dimension D).
function groupsize_shmem(ngroups::NTuple{D}, shmem_size::NTuple{D}, Np) where {D}
    # (1) Determine the total number of threads.
    # Not sure if it's worth it to define as a function of the inputs (currently unused).
    # From tests, the value of 64 seems to be optimal in various situations.
    groupsize = 64  # minimum group size should be equal to the warp size (usually 32 on CUDA and 64 on AMDGPU)
    # (2) Determine number of threads in each direction.
    # We don't really care about the Cartesian distribution of threads, since we always
    # parallelise over linear indices.
    gsizes = ntuple(_ -> 1, Val(D))
    Base.setindex(gsizes, groupsize, 1)  # = (groupsize, 1, 1, ...)
end

# Total amount of shared memory available (in bytes).
# This might be overridden in package extensions for specific backends.
available_static_shared_memory(backend::KA.Backend) = Int32(48) << 10  # 48 KiB (usual in CUDA)

# Here 0 means that the batch size will be determined automatically in order to maximise
# shared memory usage within each GPU workgroup.
const DEFAULT_GPU_BATCH_SIZE = 0

# Determine block size if using the shared-memory implementation.
# We try to make sure that the total block size (including 2M - 1 ghost cells in each direction)
# is not larger than the available shared memory. In CUDA the limit is usually 48 KiB.
# Note that the result is a compile-time constant (on Julia 1.11.1 at least).
# For this to be true, the available_static_shared_memory function should also return a
# compile-time constant (see CUDA and AMDGPU extensions for details).
@inline function block_dims_gpu_shmem(
        backend, ::Type{Z}, ::Dims{D}, ::HalfSupport{M}, ::Val{Np};
        warn = false,
    ) where {Z <: Number, D, M, Np}
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
        + 3  # buf_sm
        + D  # ishifts_sm
    )

    # (2) Shared memory required per point in a batch
    shmem_per_point = sizeof(T) * D * (
        + 2M  # window_vals
        + 1   # points_sm
    ) + (
        sizeof(Int) * D   # inds_start
    ) + (
        sizeof(Z)         # vp_sm
    )

    # Here Np is the batch size (gpu_batch_size parameter).
    # It can be DEFAULT_GPU_BATCH_SIZE (= 0) if the user didn't specify it, in which case an "optimal" value will be
    # chosen to maximise shared memory usage.
    Np_min = if Np == DEFAULT_GPU_BATCH_SIZE
        16   # minimum batch size (heuristic)
    else
        Np  # if Np was explicitly passed, make sure we leave enough space for the wanted batch size
    end

    # (3) Determine local grid size based on above values
    max_shmem_localgrid = max_shmem - const_shmem - Np_min * shmem_per_point  # maximum shared memory for local grid (single block)
    max_localgrid_length = max_shmem_localgrid ÷ sizeof(Z)  # maximum number of elements in a block
    m = floor(Int, max_localgrid_length^(1/D))  # local grid size in each direction (including ghost cells / padding)
    n = m - (2M - 1)  # exclude ghost cells
    block_dims = ntuple(_ -> n, Val(D))  # = (n, n, ...)

    if n ≤ 0
        throw(ArgumentError(
            lazy"""
            GPU shared memory size is too small for the chosen problem:
              - element type: Z = $Z
              - half-support: M = $M
              - number of dimensions: D = $D
              - spreading batch size: Np = $Np
            If possible, reduce some of these parameters, or else switch to gpu_method = :global_memory."""
        ))
    elseif warn && n < 4  # the n < 4 limit is completely empirical
        @warn lazy"""
            GPU shared memory size might be too small for the chosen problem.
            Switching to gpu_method = :global_memory will likely be faster.
            Current parameters:
              - element type: Z = $Z
              - half-support: M = $M
              - number of dimensions: D = $D
              - spreading batch size: Np = $Np
            This gives blocks of dimensions $block_dims (not including 2M - 1 = $(2M - 1) ghost cells in each direction).
            If possible, reduce some of these parameters, or else switch to gpu_method = :global_memory."""
    end

    if Np == DEFAULT_GPU_BATCH_SIZE
        # (4) Now determine Np according to the actual remaining shared memory
        shmem_left = max_shmem - const_shmem - sizeof(Z) * m^D
        Np_optimal = shmem_left ÷ shmem_per_point
        block_dims, Np_optimal
    else
        block_dims, Np
    end
end

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
