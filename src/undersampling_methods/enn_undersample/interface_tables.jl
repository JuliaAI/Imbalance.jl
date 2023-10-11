### ENN Undersampling TableTransforms Interface
# interface struct
struct ENNUndersampler{T, I<:Integer, S<:AbstractString, R<:Union{AbstractRNG, Integer}} <: Transform
    y_ind::I
    k::I
    keep_condition::S
    min_ratios::T
    force_min_ratios::Bool
    rng::R
    try_preserve_type::Bool
end

"""
Instantiate a naive ENNUndersampler table transform
"""
ENNUndersampler(
    y_ind::Integer;
    k::Integer = 5,
    keep_condition::AbstractString = "mode",
    min_ratios::Union{Nothing, AbstractFloat, Dict{T, <:AbstractFloat}} = 1.0,
    force_min_ratios::Bool = false,
    rng::Union{AbstractRNG, Integer} = default_rng(),
    try_preserve_type::Bool = true,
) where {T} = ENNUndersampler(
    y_ind,
    k,
    keep_condition,
    min_ratios,
    force_min_ratios,
    rng,
    try_preserve_type,
)

TransformsBase.isrevertible(::Type{ENNUndersampler}) = false

TransformsBase.isinvertible(::Type{ENNUndersampler}) = false

"""
Apply the ENNUndersampler transform to a table Xy

# Arguments

- `r::ENNUndersampler`: A ENNUndersampler table transform
- `Xy::AbstractTable`: A table where each row is an observation

# Returns

- `Xy_under::AbstractTable`: A table with both the original and new observations due to ENNUndersampler
"""
function TransformsBase.apply(r::ENNUndersampler, Xy)
    Xy_under = enn_undersample(
        Xy,
        r.y_ind;
        k = r.k,
        keep_condition = r.keep_condition,
        min_ratios = r.min_ratios,
        force_min_ratios = r.force_min_ratios,
        rng = r.rng,
        try_preserve_type = r.try_preserve_type,
    )
    return Xy_under, nothing
end

"""
Equivalent to `apply(r, Xy)`
"""
TransformsBase.reapply(r::ENNUndersampler, Xy, cache) = TransformsBase.apply(r, Xy)
