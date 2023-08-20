### ROSE TableTransforms Interface
## Wrap all of this in a TableTransforms module and then can use ROSE 

struct ROSE_t{T} <: TransformsBase.Transform
    y_ind::Integer
    s::AbstractFloat
    ratios::T
    rng::Union{Integer,AbstractRNG}
    try_perserve_type::Bool
end


"""
Instantiate a ROSE table transform

# Arguments

- `y_ind::Integer`: The index of the column containing the labels (integer-code) in the table
- `s::AbstractFloat=1.0`: A parameter that proportionally controls the bandwidth of the Gaussian kernel
$(DOC_RATIOS_ARGUMENT)
$(DOC_RNG_ARGUMENT)

# Returns

- `model::ROSE_t`: A SMOTE table transform that can be used like other transforms in TableTransforms.jl
"""
ROSE_t(
    y_ind::Integer;
    s::AbstractFloat = 1.0,
    ratios::Union{Nothing,AbstractFloat,Dict{T,<:AbstractFloat}} = 1.0,
    rng::Union{Integer,AbstractRNG} = 123,
    try_perserve_type::Bool = true
) where {T} = ROSE_t(y_ind, s, ratios, rng, try_perserve_type)


TransformsBase.isrevertible(::Type{ROSE_t}) = true

TransformsBase.isinvertible(::Type{ROSE_t}) = false


"""
Apply the ROSE transform to a table Xy

# Arguments

- `r::ROSE_t`: A ROSE table transform
- `Xy::AbstractTable`: A table where each row is an observation

# Returns

- `Xyover::AbstractTable`: A table with both the original and new observations due to ROSE
- `cache`: A cache that can be used to revert the oversampling
"""

function TransformsBase.apply(r::ROSE_t, Xy)
    Xyover = rose(Xy, r.y_ind; s = r.s, ratios = r.ratios, rng = r.rng, 
                  try_perserve_type = r.try_perserve_type)
    cache = rowcount(Xy)
    return Xyover, cache
end

"""
Revert the oversampling done by ROSE by removing the new observations

# Arguments

- `r::ROSE_t`: A ROSE table transform
- `Xyover::AbstractTable`: A table with both the original and new observations due to ROSE

# Returns

- `Xy::AbstractTable`: A table with only the original observations
"""
TransformsBase.revert(::ROSE_t, Xyover, cache) = revert_oversampling(Xyover, cache)

"""
Equivalent to calling `apply` but takes an extra unused argument `cache`
"""
TransformsBase.reapply(r::ROSE_t, Xy, cache) = TransformsBase.apply(r, Xy)