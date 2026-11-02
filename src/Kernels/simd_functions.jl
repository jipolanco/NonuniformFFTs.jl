# SIMD-compatible implementation of special functions.

using SIMD: SIMD

## ========================================================================================== ##
# besseli0(x)
# Adapted from Bessels.jl

besseli0_small_coefs(::Type{Float32}) = (
    1.0000000416172086f0, 0.2499995567758189f0, 0.027778554423089f0,
    0.0017355879591086688f0, 6.96204574858338f-5, 1.8959194799355527f-6,
    4.298760429567388f-8, 3.8884953805853935f-10, 1.483799986601676f-11
)
besseli0_small_coefs(::Type{Float64}) = (
    0.9999999999999998, 0.2500000000000052, 0.027777777777755364,
    0.001736111111149161, 6.94444444107536e-5, 1.9290123635366806e-6,
    3.9367592765038015e-8, 6.151201574092085e-10, 7.593827956729909e-12,
    7.596677643342155e-14, 6.255282299620455e-16, 4.470993793303175e-18,
    2.1859737023077178e-20, 2.0941557335286373e-22
)
besseli0_med_coefs(::Type{Float32}) = (
    0.3989423094522082f0, 0.049857687814784155f0, 0.028606362029333827f0,
    0.018926572151825607f0, 0.11423960556773063f0
)
besseli0_med_coefs(::Type{Float64}) = (
    0.3989422804014326, 0.04986778505064754, 0.028050628512954097,
    0.02921968830978531, 0.04466889626137549, 0.10220642174207666,
    -0.9937439085650689, 91.25330271974727, -4901.408890977662,
    199209.2752981982, -6.181516298413396e6, 1.4830278710991925e8,
    -2.7695254643719645e9, 4.0351394830842026e10, -4.5768930327229974e11,
    4.0134844243070063e12, -2.6862476523182016e13, 1.3437999451218112e14,
    -4.856333741437621e14, 1.1962791200680235e15, -1.796269414464399e15,
    1.239942074380968e15
)

function besseli0_simd(x::Union{T, SIMD.Vec{N, T} where {N}}) where {T <: Union{Float32, Float64}}
    x = abs(x)
    y1 = let
        a = x * x / 4
        muladd(a, evalpoly(a, besseli0_small_coefs(T)), 1)
    end
    y2 = let
        a = exp(x / 2)
        s = a * evalpoly(inv(x), besseli0_med_coefs(T)) / sqrt(x)
        a * s
    end
    ifelse(x < T(7.75), y1, y2)
end

## ========================================================================================== ##
# sinh(x)
# Adapted from Julia Base

@inline function two_mul(x, y)
    xy = x * y
    xy, fma(x, y, -xy)
end

# This is the same as in Base but without restricting the types to Float32 and Float64.
@inline function exthorner(x, p::NTuple{3, Any})
    hi, lo = p[lastindex(p)], zero(x)
    hi, lo = _exthorner(Val(2), x, p, hi, lo)
    hi, lo = _exthorner(Val(1), x, p, hi, lo)
    hi, lo
end

@inline function _exthorner(::Val{i}, x, p::NTuple{3, Any}, hi, lo) where {i}
    # i == 2 || i == 1 || error("unexpected index")
    pi = p[i]
    prod, err = two_mul(hi, x)
    hi = pi + prod
    lo = fma(lo, x, prod - (hi - pi) + err)
    hi, lo
end

# For Float64, use DoubleFloat scheme for extra accuracy
@inline function sinh_kernel(x::Union{Float64, SIMD.Vec{N, Float64} where {N}})
    x2, x2lo = two_mul(x, x)
    hi_order = evalpoly(x2, (8.333333333336817e-3, 1.9841269840165435e-4,
                             2.7557319381151335e-6, 2.5052096530035283e-8,
                             1.6059550718903307e-10, 7.634842144412119e-13,
                             2.9696954760355812e-15))
    hi, lo = exthorner(x2, (1.0, 0.16666666666666635, hi_order))
    muladd(x, hi, muladd(x, lo, x * x2lo * 0.16666666666666635))
end

convert_float(::Type{T}, x::S) where {T <: AbstractFloat, S <: AbstractFloat} = T(x)
convert_float(::Type{T}, x::SIMD.Vec{N, S}) where {T <: AbstractFloat, S <: AbstractFloat, N} = SIMD.Vec{N, T}(Tuple(x))

# For Float32, using Float64 is simpler, faster, and doesn't require FMA
@inline function sinh_kernel(x32::Union{Float32, SIMD.Vec{N, Float32} where {N}})
    x = convert_float(Float64, x32)
    res = evalpoly(
        x * x, (
            1.0, 0.1666666779967941, 0.008333336726447933,
            0.00019841001151414065, 2.7555538207080807e-6,
            2.5143389765825282e-8, 1.6260094552031644e-10,
        )
    )
    convert_float(Float32, res * x)
end

@inline function sinh_simd(x::Union{T, SIMD.Vec{N, T} where {N}}) where {T <: Union{Float32, Float64}}
    absx = abs(x)
    y2 = let
        E = exp(absx)
        copysign((E - 1 / E) / 2, x)
    end
    if all((Base.Math.SINH_SMALL_X(T) < absx) & (absx < Base.Math.H_LARGE_X(T)))
        return y2
    end
    y1 = sinh_kernel(x)
    if all(absx < Base.Math.H_LARGE_X(T))
        return ifelse(absx <= Base.Math.SINH_SMALL_X(T), y1, y2)
    end
    # In practice we rarely reach this point as `x` is generally not that large.
    y3 = let
        E = exp(absx / 2)
        copysign(E * E / 2, x)
    end
    ifelse(absx <= Base.Math.SINH_SMALL_X(T), y1, ifelse(absx >= Base.Math.H_LARGE_X(T), y3, y2))
end
