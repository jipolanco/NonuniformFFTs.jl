module Backends

using KernelAbstractions: KernelAbstractions as KA

export Backend, CPU, GPU

@static if isdefined(KA, :isgpu)
    # KA < 0.10 (before KA.CPU = KA.POCLBackend)
    const CPU = KA.CPU
    const GPU = KA.GPU
else
    # KA >= 0.10 (after KA.CPU = KA.POCLBackend; KA.GPU is deprecated)
    struct CPU end  # this uses the CPU path (avoids KA)
    const GPU = KA.Backend  # this includes KA.POCLBackend (runs the GPU path on the CPU)

    # Define some KA functions for our CPU backend
    KA.zeros(::CPU, ::Type{T}, args...) where {T} = zeros(T, args...)
    KA.allocate(::CPU, ::Type{T}, args...) where {T} = Array{T}(undef, args...)
end

const Backend = Union{CPU, GPU}

end
