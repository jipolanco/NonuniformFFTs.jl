module NonuniformFFTsCUDAExt

# This file contains reimplementations of spreading and interpolation kernels using CUDA.jl
# instead of KernelAbstractions.jl. Performance seems to be quite a lot better than using
# KA, especially for spreading. Note that CUDA kernels looks almost exactly the same as KA
# kernels.

using NonuniformFFTs
using NonuniformFFTs:
    Kernels, get_pointperm, get_sort_points, default_workgroupsize,
    spread_permute_kernel!, spread_from_point_naive_kernel!,
    spread_onto_arrays_gpu!, interpolate_from_arrays_gpu,
    BlockDataGPU, NullBlockData
using KernelAbstractions: KernelAbstractions as KA
using StructArrays: StructArrays, StructVector
using Static: StaticBool, True, False
using CUDA

function spread_from_point_cu!(
        us::NTuple{C},
        points::NTuple{D},
        vp::NTuple{C},
        pointperm,
        evaluate::NTuple{D},
        to_indices::NTuple{D},
    ) where {C, D}
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    N = length(points[1])

    i > N && return nothing

    j = if pointperm === nothing
        i
    else
        @inbounds pointperm[i]
    end

    x⃗ = map(xs -> @inbounds(xs[j]), points)
    v⃗ = map(v -> @inbounds(v[j]), vp)

    # Determine grid dimensions.
    Z = eltype(v⃗)
    Ns = size(first(us))
    if Z <: Complex
        @assert eltype(first(us)) <: Real      # output is a real array (but actually describes complex data)
        Ns = Base.setindex(Ns, Ns[1] >> 1, 1)  # actual number of complex elements in first dimension
    end

    # Evaluate 1D kernels.
    gs_eval = map((f, x) -> f(x), evaluate, x⃗)

    # Determine indices to write in `u` arrays.
    indvals = map(to_indices, gs_eval, Ns) do f, gdata, N
        f(gdata.i, N) => gdata.values
    end

    spread_onto_arrays_gpu!(us, indvals, v⃗)

    nothing
end

# We assume all arrays are already on the GPU.
function NonuniformFFTs.exec_spreading_kernel!(::CUDABackend, us_real::NTuple, points::NTuple, args...)
    ndrange = size(points[1])
    nblocks = cld(ndrange[1], 512)
    @cuda threads=512 blocks=nblocks spread_from_point_cu!(us_real, points, args...)
    nothing
end

function interpolate_to_point_cu!(
        vp::NTuple{C},
        points::NTuple{D},
        us::NTuple{C},
        pointperm,
        Δxs::NTuple{D},           # grid step in each direction (oversampled grid)
        evaluate::NTuple{D, <:Function},  # can't be marked Const for some reason
        to_indices::NTuple{D, <:Function},
    ) where {C, D}
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    N = length(points[1])

    i > N && return nothing

    j = if pointperm === nothing
        i
    else
        @inbounds pointperm[i]
    end

    x⃗ = map(xs -> @inbounds(xs[j]), points)

    # Determine grid dimensions.
    # Unlike in spreading, here `us` can be made of arrays of complex numbers, because we
    # don't perform atomic operations. This is why the code is simpler here.
    Ns = size(first(us))  # grid dimensions

    # Evaluate 1D kernels.
    gs_eval = map((f, x) -> f(x), evaluate, x⃗)

    # Determine indices to load from `u` arrays.
    indvals = map(to_indices, gs_eval, Ns, Δxs) do f, gdata, N, Δx
        vals = gdata.values .* Δx
        f(gdata.i, N) => vals
    end

    v⃗ = interpolate_from_arrays_gpu(us, indvals)

    for (dst, v) ∈ zip(vp, v⃗)
        @inbounds dst[j] = v
    end

    nothing
end

function NonuniformFFTs.exec_interpolation_kernel!(::CUDABackend, vp::NTuple, points::NTuple, args...)
    ndrange = size(points[1])
    nblocks = cld(ndrange[1], 512)
    @cuda threads=512 blocks=nblocks interpolate_to_point_cu!(vp, points, args...)
    nothing
end

end
