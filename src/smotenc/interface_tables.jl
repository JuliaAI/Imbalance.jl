### SMOTENC TableTransforms Interface

struct SMOTENC{T,R<:Union{Integer,AbstractRNG}} <: TransformsBase.Transform
    y_ind::Integer
    k::Integer
    ratios::T
    rng::R
    try_perserve_type::Bool
end


TransformsBase.isrevertible(::Type{SMOTENC}) = true
TransformsBase.isinvertible(::Type{SMOTENC}) = false

"""
Instantiate a SMOTENC table transform

# Arguments

- `y_ind::Integer`: The index of the column containing the labels (integer-code) in the table
- `k::Integer`: Number of nearest neighbors to consider in the SMOTENC algorithm. 
    Should be within the range `[1, size(X, 1) - 1]` else set to the nearest of these two values.
$((COMMON_DOCS["RATIOS"]))
$((COMMON_DOCS["RNG"]))

# Returns

- `model::SMOTENC`: A SMOTENC table transform that can be used like other transforms in TableTransforms.jl

"""
SMOTENC(
    y_ind::Integer;
    k::Integer = 5,
    ratios::Union{Nothing,AbstractFloat,Dict{T,<:AbstractFloat}} = 1.0,
    rng::Union{Integer,AbstractRNG} = 123,
    try_perserve_type::Bool=true
) where {T} = SMOTENC(y_ind, k, ratios, rng, try_perserve_type)


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
    Xyover = smotenc(Xy, s.y_ind; k = s.k, ratios = s.ratios, rng = s.rng,
                        try_perserve_type = s.try_perserve_type)
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
