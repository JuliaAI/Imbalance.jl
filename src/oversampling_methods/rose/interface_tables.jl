### ROSE TableTransforms Interface
## Wrap all of this in a TableTransforms module and then can use ROSE 

struct ROSE{T,R<:Union{Integer,AbstractRNG}, I<:Integer, F<:AbstractFloat} <: TransformsBase.Transform
    y_ind::I
    s::F
    ratios::T
    rng::R
    try_preserve_type::Bool
end




"""
Instantiate a ROSE table transform
"""
ROSE(
    y_ind::Integer;
    s::AbstractFloat = 1.0,
    ratios::Union{Nothing,AbstractFloat,Dict{T,<:AbstractFloat}} = 1.0,
    rng::Union{Integer,AbstractRNG} = 123,
    try_preserve_type::Bool = true
) where {T} = ROSE(y_ind, s, ratios, rng, try_preserve_type)


TransformsBase.isrevertible(::Type{ROSE}) = true

TransformsBase.isinvertible(::Type{ROSE}) = false


"""
Apply the ROSE transform to a table Xy

# Arguments

- `r::ROSE`: A ROSE table transform
- `Xy::AbstractTable`: A table where each row is an observation

# Returns

- `Xyover::AbstractTable`: A table with both the original and new observations due to ROSE
- `cache`: A cache that can be used to revert the oversampling
"""

function TransformsBase.apply(r::ROSE, Xy)
    Xyover = rose(Xy, r.y_ind; s = r.s, ratios = r.ratios, rng = r.rng, 
                  try_preserve_type = r.try_preserve_type)
    cache = rowcount(Xy)
    return Xyover, cache
end

"""
Revert the oversampling done by ROSE by removing the new observations

# Arguments

- `r::ROSE`: A ROSE table transform
- `Xyover::AbstractTable`: A table with both the original and new observations due to ROSE

# Returns

- `Xy::AbstractTable`: A table with only the original observations
"""
TransformsBase.revert(::ROSE, Xyover, cache) = revert_oversampling(Xyover, cache)

"""
Equivalent to calling `apply` but takes an extra unused argument `cache`
"""
TransformsBase.reapply(r::ROSE, Xy, cache) = TransformsBase.apply(r, Xy)
