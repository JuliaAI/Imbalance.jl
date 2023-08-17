### SMOTE TableTransforms Interface

struct SMOTE_t{T} <: TransformsBase.Transform
    y_ind::Integer
    k::Integer
    ratios::T
    rng::Union{Integer,AbstractRNG}
end

TransformsBase.isrevertible(::Type{SMOTE_t}) = true
TransformsBase.isinvertible(::Type{SMOTE_t}) = false

"""
Instantiate a SMOTE table transform

# Arguments

- `y_ind::Integer`: The index of the column containing the labels (integer-code) in the table
- `k::Integer`: Number of nearest neighbors to consider in the SMOTE algorithm. 
    Should be within the range `[1, size(X, 1) - 1]` else set to the nearest of these two values.
$(DOC_RATIOS_ARGUMENT)
$(DOC_RNG_ARGUMENT)

# Returns

- `model::SMOTE_t`: A SMOTE table transform that can be used like other transforms in TableTransforms.jl

"""
SMOTE_t(
    y_ind::Integer;
    k::Integer = 5,
    ratios::Union{Nothing,AbstractFloat,Dict{T,<:AbstractFloat}} = nothing,
    rng::Union{Integer,AbstractRNG} = 123,
) where {T} = SMOTE_t(y_ind, k, ratios, rng)


"""
Apply the SMOTE transform to a table Xy

# Arguments

- `s::SMOTE_t`: A SMOTE table transform

- `Xy::AbstractTable`: A table where each row is an observation

# Returns

- `Xyover::AbstractTable`: A table with both the original and new observations due to SMOTE
- `cache`: A cache that can be used to revert the oversampling
"""
function TransformsBase.apply(s::SMOTE_t, Xy)
    Xyover = smote(Xy, s.y_ind; k = s.k, ratios = s.ratios, rng = s.rng)
    cache = rowcount(Xy)
    return Xyover, cache
end


"""
Revert the oversampling done by SMOTE by removing the new observations

# Arguments

- `s::SMOTE_t`: A SMOTE table transform
- `Xyover::AbstractTable`: A table with both the original and new observations due to SMOTE
- `cache`: cache returned from `apply`

# Returns

- `Xy::AbstractTable`: A table with the original observations only
"""
TransformsBase.revert(::SMOTE_t, Xyover, cache) = revert_oversampling(Xyover, cache)

"""
Equivalent to `apply(s, Xy)`
"""
TransformsBase.reapply(s::SMOTE_t, Xy, cache) = TransformsBase.apply(s, Xy)


