### Random Oversample TableTransforms Interface
# interface struct
struct RandomOversampler{T,R<:Union{Integer,AbstractRNG}} <: Transform
    y_ind::Integer
    ratios::T
    rng::R
    try_perserve_type::Bool
end

"""
Instantiate a naive RandomOversampler table transform

# Arguments

- `y_ind::Integer`: The index of the column containing the labels (integer-code) in the table
$((COMMON_DOCS["RATIOS"]))
$((COMMON_DOCS["RNG"]))

# Returns

- `model::RandomOversampler`: A SMOTE table transform that can be used like other transforms in TableTransforms.jl
"""
RandomOversampler(
    y_ind::Integer;
    ratios::Union{Nothing,AbstractFloat,Dict{T,<:AbstractFloat}} = 1.0,
    rng::Union{Integer,AbstractRNG} = 123,
    try_perserve_type::Bool = true,
) where {T} = RandomOversampler(y_ind, ratios, rng, try_perserve_type)


TransformsBase.isrevertible(::Type{RandomOversampler}) = true

TransformsBase.isinvertible(::Type{RandomOversampler}) = false

"""
Apply the RandomOversampler transform to a table Xy

# Arguments

- `r::RandomOversampler`: A RandomOversampler table transform
- `Xy::AbstractTable`: A table where each row is an observation

# Returns

- `Xyover::AbstractTable`: A table with both the original and new observations due to RandomOversampler
- `cache`: A cache that can be used to revert the oversampling
"""
function TransformsBase.apply(r::RandomOversampler, Xy)
    Xyover = random_oversample(Xy, r.y_ind; ratios = r.ratios, rng = r.rng,
                               try_perserve_type = r.try_perserve_type)
    # so that we can revert later by removing the new observations:
    cache = rowcount(Xy)
    return Xyover, cache
end

"""
Revert the oversampling done by RandomOversampler by removing the new observations

# Arguments

- `r::RandomOversampler`: A RandomOversampler table transform
- `Xyover::AbstractTable`: A table with both the original and new observations due to RandomOversampler

# Returns

- `Xy::AbstractTable`: A table with only the original observations
"""
TransformsBase.revert(::RandomOversampler, Xyover, cache) =
    revert_oversampling(Xyover, cache)

"""
Equivalent to `apply(r, Xy)`
"""
TransformsBase.reapply(r::RandomOversampler, Xy, cache) = TransformsBase.apply(r, Xy)
