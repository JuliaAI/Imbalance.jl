### RandomWalkOversampler TableTransforms Interface

struct RandomWalkOversampler{T, R<:Union{Integer,AbstractRNG}, I<:Integer} <: TransformsBase.Transform
    y_ind::I
    ratios::T
    rng::R
    try_perserve_type::Bool
end


TransformsBase.isrevertible(::Type{RandomWalkOversampler}) = true
TransformsBase.isinvertible(::Type{RandomWalkOversampler}) = false

"""
Instantiate a `RandomWalkOversampler` table transform
"""
RandomWalkOversampler(
    y_ind::Integer;
    ratios::Union{Nothing,AbstractFloat,Dict{T,<:AbstractFloat}} = 1.0,
    rng::Union{Integer,AbstractRNG} = default_rng(),
    try_perserve_type::Bool=true
) where {T} = RandomWalkOversampler(y_ind, ratios, rng, try_perserve_type)


"""
Apply the `RandomWalkOversampler` transform to a table Xy

# Arguments

- `s::RandomWalkOversampler`: A RandomWalkOversampler table transform

- `Xy::AbstractTable`: A table where each row is an observation

# Returns

- `Xyover::AbstractTable`: A table with both the original and new observations due to RandomWalkOversampler
- `cache`: A cache that can be used to revert the oversampling
"""
function TransformsBase.apply(s::RandomWalkOversampler, Xy)
    Xyover = random_walk_oversample(Xy, s.y_ind; ratios = s.ratios, rng = s.rng,
                        try_perserve_type = s.try_perserve_type)
    cache = rowcount(Xy)
    return Xyover, cache
end


"""
Revert the oversampling done by RandomWalkOversampler by removing the new observations

# Arguments

- `s::RandomWalkOversampler`: A RandomWalkOversampler table transform
- `Xyover::AbstractTable`: A table with both the original and new observations due to RandomWalkOversampler
- `cache`: cache returned from `apply`

# Returns

- `Xy::AbstractTable`: A table with the original observations only
"""
TransformsBase.revert(::RandomWalkOversampler, Xyover, cache) = revert_oversampling(Xyover, cache)

"""
Equivalent to `apply(s, Xy)`
"""
TransformsBase.reapply(s::RandomWalkOversampler, Xy, cache) = TransformsBase.apply(s, Xy)