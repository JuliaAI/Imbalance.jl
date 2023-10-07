### SMOTENC TableTransforms Interface

struct SMOTENC{T,R<:Union{Integer,AbstractRNG}, S<:AbstractString, I<:Integer} <: TransformsBase.Transform
    y_ind::I
    k::I
    ratios::T
    knn_tree::S
    rng::R
    try_preserve_type::Bool
end


TransformsBase.isrevertible(::Type{SMOTENC}) = true
TransformsBase.isinvertible(::Type{SMOTENC}) = false

"""
Instantiate a SMOTENC table transform
"""
SMOTENC(
    y_ind::Integer;
    k::Integer = 5,
    ratios::Union{Nothing,AbstractFloat,Dict{T,<:AbstractFloat}} = 1.0,
    knn_tree::AbstractString = "Brute",
    rng::Union{Integer,AbstractRNG} = 123,
    try_preserve_type::Bool=true
) where {T} = SMOTENC(y_ind, k, ratios, knn_tree, rng, try_preserve_type)


"""
Apply the SMOTENC transform to a table Xy

# Arguments

- `s::SMOTENC`: A SMOTENC table transform

- `Xy::AbstractTable`: A table where each row is an observation

# Returns

- `Xyover::AbstractTable`: A table with both the original and new observations due to SMOTENC
- `cache`: A cache that can be used to revert the oversampling
"""
function TransformsBase.apply(s::SMOTENC, Xy)
    Xyover = smotenc(Xy, s.y_ind; k = s.k, ratios = s.ratios, knn_tree=s.knn_tree, rng = s.rng,
                        try_preserve_type = s.try_preserve_type)
    cache = rowcount(Xy)
    return Xyover, cache
end


"""
Revert the oversampling done by SMOTENC by removing the new observations

# Arguments

- `s::SMOTENC`: A SMOTENC table transform
- `Xyover::AbstractTable`: A table with both the original and new observations due to SMOTENC
- `cache`: cache returned from `apply`

# Returns

- `Xy::AbstractTable`: A table with the original observations only
"""
TransformsBase.revert(::SMOTENC, Xyover, cache) = revert_oversampling(Xyover, cache)

"""
Equivalent to `apply(s, Xy)`
"""
TransformsBase.reapply(s::SMOTENC, Xy, cache) = TransformsBase.apply(s, Xy)
