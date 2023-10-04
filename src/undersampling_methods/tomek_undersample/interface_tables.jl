### Tomek Undersampling TableTransforms Interface
# interface struct
struct TomekUndersampler{T, R <: Union{Integer, AbstractRNG}, I<:Integer} <: Transform
    y_ind::I
    min_ratios::T
    force_min_ratios::Bool
    rng::R
    try_perserve_type::Bool
end

"""
Instantiate a TomekUndersampler table transform
"""
TomekUndersampler(
    y_ind::Integer;
    min_ratios::Union{Nothing, AbstractFloat, Dict{T, <:AbstractFloat}} = 1.0,
    force_min_ratios::Bool = false,
    rng::Union{Integer, AbstractRNG} = 123,
    try_perserve_type::Bool = true,
) where {T} = TomekUndersampler(y_ind, min_ratios, force_min_ratios, rng, try_perserve_type)

TransformsBase.isrevertible(::Type{TomekUndersampler}) = false

TransformsBase.isinvertible(::Type{TomekUndersampler}) = false

"""
Apply the TomekUndersampler transform to a table Xy

# Arguments

- `r::TomekUndersampler`: A TomekUndersampler table transform
- `Xy::AbstractTable`: A table where each row is an observation

# Returns

- `Xy_under::AbstractTable`: A table with both the original and new observations due to TomekUndersampler
"""
function TransformsBase.apply(r::TomekUndersampler, Xy)
    Xy_under = tomek_undersample(
        Xy,
        r.y_ind;
        min_ratios = r.min_ratios,
        force_min_ratios = r.force_min_ratios,
        rng = r.rng,
        try_perserve_type = r.try_perserve_type,
    )
    return Xy_under, nothing
end

"""
Equivalent to `apply(r, Xy)`
"""
TransformsBase.reapply(r::TomekUndersampler, Xy, cache) = TransformsBase.apply(r, Xy)
