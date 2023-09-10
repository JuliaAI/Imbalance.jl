### Cluster Undersampling TableTransforms Interface
# interface struct
struct ClusterUndersampler{T} <: Transform
    y_ind::Integer
    mode::String
    ratios::T
    maxiter::Integer
    rng::Integer
    try_perserve_type::Bool
end

"""
Instantiate a naive ClusterUndersampler table transform

# Arguments

- `y_ind::Integer`: The index of the column containing the labels in the table

- `mode::String="nearest`: If `center` then the undersampled data will consist of the centriods of 
    each cluster found. Meanwhile, if `nearest` then it will consist of the nearest neighbor of each centroid.

$(COMMON_DOCS["RATIOS-UNDERSAMPLE"])

- `maxiter::Integer=100`: Maximum number of iterations to run K-means

- `rng::Integer=42`: Random number generator seed. Must be an integer.

$(COMMON_DOCS["TRY_PERSERVE_TYPE"])

# Returns

- `model::ClusterUndersampler`: A Cluster Undersampling table transform that can be 
    used like other transforms in TableTransforms.jl
"""
ClusterUndersampler(
    y_ind::Integer;
    mode::String = "nearest", ratios::Union{Nothing,AbstractFloat,Dict{T,<:AbstractFloat}} = 1.0,
    maxiter::Integer = 100, rng::Integer = 42, 
    try_perserve_type::Bool = true
) where {T} = ClusterUndersampler(y_ind, mode, ratios, maxiter, rng, try_perserve_type)


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
    Xy_under = cluster_undersample(Xy, r.y_ind; mode = r.mode, ratios = r.ratios, maxiter=r.maxiter,
                                     rng = r.rng, try_perserve_type = r.try_perserve_type)
    return Xy_under, nothing
end


"""
Equivalent to `apply(r, Xy)`
"""
TransformsBase.reapply(r::ClusterUndersampler, Xy, cache) = TransformsBase.apply(r, Xy)
