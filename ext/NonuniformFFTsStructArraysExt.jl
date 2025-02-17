# We keep this for backwards compatibility, as set_points! used to be implemented using
# StructArrays for storing point locations.
module NonuniformFFTsStructArraysExt

using NonuniformFFTs
using StructArrays: StructArrays, StructVector

NonuniformFFTs.set_points!(p::PlanNUFFT, xp::StructVector; kwargs...) =
    set_points!(p, StructArrays.components(xp); kwargs...)

end
