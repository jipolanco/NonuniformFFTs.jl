module NonuniformFFTsOpenCLExt

using NonuniformFFTs
using OpenCL: CLDeviceArray
using Atomix: Atomix

# Workaround atomic add with floats often not supported by OpenCL.
@inline function _opencl_atomic_add_float!(u::CLDeviceArray{T}, v::T, inds::Tuple) where {T <: AbstractFloat}
    # @inbounds Atomix.@atomic :monotonic u[inds...] += v  # ideally we'd like to do this, but generally it doesn't work with OpenCL
    @inbounds old = u[inds...]  # current value
    success = false
    @inbounds while !success
        new = old + v  # new value we want to write -- here `old` is the latest read value
        (; success, old) = Atomix.@atomicreplace :monotonic u[js...] old => new  # try to write new value, if `old` one hasn't changed since
    new = old + v  # new value we want to write -- here `old` is the latest read value
    @inbounds while true
        (; success, old) = Atomix.@atomicreplace :monotonic u[inds...] old => new  # try to write new value, if `old` one hasn't changed since
        success && break
        new = old + v
    end
    new
end

@inline function NonuniformFFTs._atomic_add!(u::CLDeviceArray{T}, v::T, inds::Tuple) where {T <: Real}
    _opencl_atomic_add!(u, v, inds)
end

@inline function NonuniformFFTs._atomic_add!(u::CLDeviceArray{T}, v::Complex{T}, inds::Tuple) where {T <: Real}
    a = _opencl_atomic_add!(u, real(v), (1, inds...))
    b = _opencl_atomic_add!(u, imag(v), (2, inds...))
    Complex(a, b)
end

end
