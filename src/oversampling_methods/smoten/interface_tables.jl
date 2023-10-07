### SMOTEN TableTransforms Interface

struct SMOTEN{T,R<:Union{Integer,AbstractRNG}, I<:Integer} <: TransformsBase.Transform
    y_ind::I
    k::I
    ratios::T
    rng::R
    try_preserve_type::Bool
end



TransformsBase.isrevertible(::Type{SMOTEN}) = true
TransformsBase.isinvertible(::Type{SMOTEN}) = false

"""
Instantiate a SMOTEN table transform
"""
SMOTEN(
    y_ind::Integer;
    k::Integer = 5,
    ratios::Union{Nothing,AbstractFloat,Dict{T,<:AbstractFloat}} = 1.0,
    rng::Union{Integer,AbstractRNG} = 123, try_preserve_type::Bool=true
) where {T} = SMOTEN(y_ind, k, ratios, rng, try_preserve_type)


"""
Apply the SMOTEN transform to a table Xy

# Arguments

- `s::SMOTEN`: A SMOTEN table transform

- `Xy::AbstractTable`: A table where each row is an observation

# Returns

- `Xyover::AbstractTable`: A table with both the original and new observations due to SMOTEN
- `cache`: A cache that can be used to revert the oversampling
"""
function TransformsBase.apply(s::SMOTEN, Xy)
    Xyover = smoten(Xy, s.y_ind; k = s.k, ratios = s.ratios, rng = s.rng, 
                    try_preserve_type = s.try_preserve_type)
    cache = rowcount(Xy)
    return Xyover, cache
end


"""
Revert the oversampling done by SMOTEN by removing the new observations

# Arguments

- `s::SMOTEN`: A SMOTEN table transform
- `Xyover::AbstractTable`: A table with both the original and new observations due to SMOTEN
- `cache`: cache returned from `apply`

# Returns

- `Xy::AbstractTable`: A table with the original observations only
"""
TransformsBase.revert(::SMOTEN, Xyover, cache) = revert_oversampling(Xyover, cache)

"""
Equivalent to `apply(s, Xy)`
"""
TransformsBase.reapply(s::SMOTEN, Xy, cache) = TransformsBase.apply(s, Xy)
