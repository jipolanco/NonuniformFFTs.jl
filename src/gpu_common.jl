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

# This is called from the global memory (naive) implementation of spreading and interpolation kernels.
@inline function get_inds_vals_gpu(gs::NTuple{D}, points::NTuple{D}, Ns::NTuple{D}, j::Integer) where {D}
    ntuple(Val(D)) do n
        @inline
        get_inds_vals_gpu(gs[n], points[n], Ns[n], j)
    end
end

@inline function get_inds_vals_gpu(g::AbstractKernelData, points::AbstractVector, N::Integer, j::Integer)
    x = @inbounds points[j]
    gdata = Kernels.evaluate_kernel(g, x)
    vals = gdata.values    # kernel values
    M = Kernels.half_support(g)
    i₀ = gdata.i - M  # active region is (i₀ + 1):(i₀ + 2M) (up to periodic wrapping)
    i₀ = ifelse(i₀ < 0, i₀ + N, i₀)  # make sure i₀ ≥ 0
    i₀ => vals
end
