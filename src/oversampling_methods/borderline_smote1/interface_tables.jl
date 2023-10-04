### BorderlineSMOTE1 TableTransforms Interface

struct BorderlineSMOTE1{T,R<:Union{Integer,AbstractRNG}, I<:Integer} <: TransformsBase.Transform
    y_ind::I
    m::I
    k::I
    ratios::T
    rng::R
    try_perserve_type::Bool
    verbosity::I
end


TransformsBase.isrevertible(::Type{BorderlineSMOTE1}) = true
TransformsBase.isinvertible(::Type{BorderlineSMOTE1}) = false

"""
Instantiate a BorderlineSMOTE1 table transform
"""
BorderlineSMOTE1(
    y_ind::Integer;
    m::Integer=5,
    k::Integer = 5,
    ratios::Union{Nothing,AbstractFloat,Dict{T,<:AbstractFloat}} = 1.0,
    rng::Union{Integer,AbstractRNG} = default_rng(), try_perserve_type::Bool=true, verbosity::Integer=1
) where {T} = BorderlineSMOTE1(y_ind, m, k, ratios, rng, try_perserve_type, verbosity)


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
                   try_perserve_type = s.try_perserve_type, verbosity=s.verbosity)
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
