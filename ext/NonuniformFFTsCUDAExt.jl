module NonuniformFFTsCUDAExt

using NonuniformFFTs
using CUDA

@inline function NonuniformFFTs._atomic_add!(u::CuDeviceArray{T}, v::T, inds::Tuple) where {T <: Real}
    @inbounds CUDA.@atomic u[inds...] += v
end

@inline function NonuniformFFTs._atomic_add!(u::CuDeviceArray{T}, v::Complex{T}, inds::Tuple) where {T <: Real}
    @inbounds begin
        i₁ = 2 * (inds[1] - 1)  # convert from logical index (equivalent complex array) to memory index (real array)
        itail = Base.tail(inds)
        CUDA.@atomic u[i₁ + 1, itail...] += real(v)
        CUDA.@atomic u[i₁ + 2, itail...] += imag(v)
    end
    nothing
end

end
