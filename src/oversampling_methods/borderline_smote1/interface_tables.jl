### BorderlineSMOTE1 TableTransforms Interface

struct BorderlineSMOTE1{T,R<:Union{Integer,AbstractRNG}, I<:Integer} <: TransformsBase.Transform
    y_ind::I
    m::I
    k::I
    ratios::T
    rng::R
    try_perserve_type::Bool
end


TransformsBase.isrevertible(::Type{BorderlineSMOTE1}) = true
TransformsBase.isinvertible(::Type{BorderlineSMOTE1}) = false

"""
Instantiate a BorderlineSMOTE1 table transform

# Arguments

- `y_ind::Integer`: The index of the column containing the labels (integer-code) in the table

- `m::Integer=5`: The number of neighbors to consider while checking the BorderlineSMOTE1 condition. In this, a point may participate
    in oversampling iff the number of neighbors that belong to its class is in \$(0, m/2]\$. Should be within the range 
   `0 < m < N` where N is the number of observations in the data. It will be automatically set to `N-1` if `N ≤ m`.

$(COMMON_DOCS["K"])

$((COMMON_DOCS["RATIOS"]))
$((COMMON_DOCS["RNG"]))

# Returns

- `model::BorderlineSMOTE1`: A BorderlineSMOTE1 table transform that can be used like other transforms in TableTransforms.jl

"""
BorderlineSMOTE1(
    y_ind::Integer;
    m::Integer=5,
    k::Integer = 5,
    ratios::Union{Nothing,AbstractFloat,Dict{T,<:AbstractFloat}} = 1.0,
    rng::Union{Integer,AbstractRNG} = default_rng(), try_perserve_type::Bool=true
) where {T} = BorderlineSMOTE1(y_ind, m, k, ratios, rng, try_perserve_type)


"""
Apply the BorderlineSMOTE1 transform to a table Xy

# Arguments

- `s::BorderlineSMOTE1`: A BorderlineSMOTE1 table transform

- `Xy::AbstractTable`: A table where each row is an observation

# Returns

- `Xyover::AbstractTable`: A table with both the original and new observations due to BorderlineSMOTE1
- `cache`: A cache that can be used to revert the oversampling
"""
function TransformsBase.apply(s::BorderlineSMOTE1, Xy)
    Xyover = borderline_smote1(Xy, s.y_ind; m = s.m, k = s.k, ratios = s.ratios, rng = s.rng, 
                   try_perserve_type = s.try_perserve_type)
    cache = rowcount(Xy)
    return Xyover, cache
end


"""
Revert the oversampling done by BorderlineSMOTE1 by removing the new observations

# Arguments

- `s::BorderlineSMOTE1`: A BorderlineSMOTE1 table transform
- `Xyover::AbstractTable`: A table with both the original and new observations due to BorderlineSMOTE1
- `cache`: cache returned from `apply`

# Returns

- `Xy::AbstractTable`: A table with the original observations only
"""
TransformsBase.revert(::BorderlineSMOTE1, Xyover, cache) = revert_oversampling(Xyover, cache)

"""
Equivalent to `apply(s, Xy)`
"""
TransformsBase.reapply(s::BorderlineSMOTE1, Xy, cache) = TransformsBase.apply(s, Xy)
