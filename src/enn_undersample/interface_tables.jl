### ENN Undersampling TableTransforms Interface
# interface struct
struct ENNUndersampler{T} <: Transform
    y_ind::Integer
    k::Integer
    keep_condition::AbstractString
    min_ratios::T
    force_min_ratios::Bool
    rng::Integer
    try_perserve_type::Bool
end

"""
Instantiate a naive ENNUndersampler table transform

# Arguments

- `y_ind::Integer`: The index of the column containing the labels in the table

$(COMMON_DOCS["K"])

- `keep_condition="mode"`: The condition that leads to cleaning a point upon violation. Takes one of `"exists"`, `"mode"`, `"only mode"` and `"all"`
    - `"exists"`: the point has at least one neighbor from the same class
    - `"mode"`: the class of the point is one of the most frequent classes of the neighbors (there may be many)
    - `"only mode"`: the class of the point is the single most frequent class of the neighbors
    - `"all"`: the class of the point is the same as all the neighbors

$(COMMON_DOCS["MIN-RATIOS-UNDERSAMPLE"])

$(COMMON_DOCS["FORCE-MIN-RATIOS"])

$(COMMON_DOCS["RNG"])

$(COMMON_DOCS["TRY_PERSERVE_TYPE"])

# Returns

- `model::ENNUndersampler`: A ENN Undersampling table transform that can be 
    used like other transforms in TableTransforms.jl
"""
ENNUndersampler(
    y_ind::Integer;
    k::Integer = 5,
    keep_condition::AbstractString = "mode",
    min_ratios::Union{Nothing, AbstractFloat, Dict{T, <:AbstractFloat}} = 1.0,
    force_min_ratios::Bool = false,
    rng::Integer = 42,
    try_perserve_type::Bool = true,
) where {T} = ENNUndersampler(
    y_ind,
    k,
    keep_condition,
    min_ratios,
    force_min_ratios,
    rng,
    try_perserve_type,
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
        try_perserve_type = r.try_perserve_type,
    )
    return Xy_under, nothing
end

"""
Equivalent to `apply(r, Xy)`
"""
TransformsBase.reapply(r::ENNUndersampler, Xy, cache) = TransformsBase.apply(r, Xy)
