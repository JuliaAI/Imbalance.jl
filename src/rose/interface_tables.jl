### ROSE TableTransforms Interface
## Wrap all of this in a TableTransforms module and then can use ROSE 

struct ROSE_t{T} <: TransformsBase.Transform
    y_ind::Integer
    s::AbstractFloat
    ratios::T
    rng::Union{Integer,AbstractRNG}
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
    ratios::Union{Nothing,AbstractFloat,Dict{T,<:AbstractFloat}} = nothing,
    rng::Union{Integer,AbstractRNG} = 123,
) where {T} = ROSE_t(y_ind, s, ratios, rng)


TransformsBase.isrevertible(::Type{ROSE_t}) = true

TransformsBase.isinvertible(::Type{ROSE_t}) = false

TransformsBase.assertions(::Type{ROSE_t}) =
    [rose -> @assert rose.s >= 0.0 "s parameter in ROSE must be non-negative"]
# for consistency, may use throw error (or warn and set 0)

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
    for assertion in TransformsBase.assertions(ROSE_t)
        assertion(r)
    end
    Xyover = rose(Xy, r.y_ind; s = r.s, ratios = r.ratios, rng = r.rng)
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

