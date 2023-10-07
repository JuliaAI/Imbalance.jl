### Cluster Undersampling TableTransforms Interface
# interface struct
struct ClusterUndersampler{T, I<:Integer, S<:AbstractString, R<:Union{AbstractRNG, Integer}} <: Transform
    y_ind::I
    mode::S
    ratios::T
    maxiter::I
    rng::R
    try_preserve_type::Bool
end

"""
Instantiate a naive ClusterUndersampler table transform
"""
ClusterUndersampler(
    y_ind::Integer;
    mode::AbstractString = "nearest",
    ratios::Union{Nothing, AbstractFloat, Dict{T, <:AbstractFloat}} = 1.0,
    maxiter::Integer = 100,
    rng::Union{Integer, AbstractRNG} = default_rng(),
    try_preserve_type::Bool = true,
) where {T} = ClusterUndersampler(y_ind, mode, ratios, maxiter, rng, try_preserve_type)

TransformsBase.isrevertible(::Type{ClusterUndersampler}) = false

TransformsBase.isinvertible(::Type{ClusterUndersampler}) = false

"""
Apply the ClusterUndersampler transform to a table Xy

# Arguments

- `r::ClusterUndersampler`: A ClusterUndersampler table transform
- `Xy::AbstractTable`: A table where each row is an observation

# Returns

- `Xy_under::AbstractTable`: A table with both the original and new observations due to ClusterUndersampler
"""
function TransformsBase.apply(r::ClusterUndersampler, Xy)
    Xy_under = cluster_undersample(
        Xy,
        r.y_ind;
        mode = r.mode,
        ratios = r.ratios,
        maxiter = r.maxiter,
        rng = r.rng,
        try_preserve_type = r.try_preserve_type,
    )
    return Xy_under, nothing
end

"""
Equivalent to `apply(r, Xy)`
"""
TransformsBase.reapply(r::ClusterUndersampler, Xy, cache) = TransformsBase.apply(r, Xy)
