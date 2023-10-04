### Random Undersampling TableTransforms Interface
# interface struct
struct RandomUndersampler{T, R <: Union{Integer, AbstractRNG}, I<:Integer} <: Transform
    y_ind::I
    ratios::T
    rng::R
    try_perserve_type::Bool
end

"""
Instantiate a naive RandomUndersampler table transform
"""
RandomUndersampler(
    y_ind::Integer;
    ratios::Union{Nothing, AbstractFloat, Dict{T, <:AbstractFloat}} = 1.0,
    rng::Union{Integer, AbstractRNG} = 123,
    try_perserve_type::Bool = true,
) where {T} = RandomUndersampler(y_ind, ratios, rng, try_perserve_type)

TransformsBase.isrevertible(::Type{RandomUndersampler}) = false

TransformsBase.isinvertible(::Type{RandomUndersampler}) = false

"""
Apply the RandomUndersampler transform to a table Xy

# Arguments

- `r::RandomUndersampler`: A RandomUndersampler table transform
- `Xy::AbstractTable`: A table where each row is an observation

# Returns

- `Xy_under::AbstractTable`: A table with both the original and new observations due to RandomUndersampler
"""
function TransformsBase.apply(r::RandomUndersampler, Xy)
    Xy_under = random_undersample(
        Xy,
        r.y_ind;
        ratios = r.ratios,
        rng = r.rng,
        try_perserve_type = r.try_perserve_type,
    )
    return Xy_under, nothing
end

"""
Equivalent to `apply(r, Xy)`
"""
TransformsBase.reapply(r::RandomUndersampler, Xy, cache) = TransformsBase.apply(r, Xy)
