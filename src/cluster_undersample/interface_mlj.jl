
### ClusterUndersampler with MLJ Interface
# interface struct
mutable struct ClusterUndersampler{T} <: Static
    mode::AbstractString
    ratios::T
    maxiter::Integer
    rng::Integer
    try_perserve_type::Bool
end;

"""
Initiate a cluster undersampling model with the given hyper-parameters.
"""
function ClusterUndersampler(;
    mode::AbstractString = "nearest",
    ratios::Union{Nothing, AbstractFloat, Dict{T, <:AbstractFloat}} = 1.0,
    maxiter::Integer = 100,
    rng::Integer = 42,
    try_perserve_type::Bool = true,
) where {T}
    model = ClusterUndersampler(mode, ratios, maxiter, rng, try_perserve_type)
    return model
end

"""
Undersample data X, y 
"""
function MMI.transform(r::ClusterUndersampler, _, X, y)
    return cluster_undersample(
        X,
        y;
        mode = r.mode,
        ratios = r.ratios,
        maxiter = r.maxiter,
        rng = r.rng,
        try_perserve_type = r.try_perserve_type,
    )
end

MMI.metadata_pkg(
    ClusterUndersampler,
    name = "Imbalance",
    package_uuid = "c709b415-507b-45b7-9a3d-1767c89fde68",
    package_url = "https://github.com/JuliaAI/Imbalance.jl",
    is_pure_julia = true,
)

MMI.metadata_model(
    ClusterUndersampler,
    input_scitype = Union{Table(Continuous), AbstractMatrix{Continuous}},
    output_scitype = Union{Table(Continuous), AbstractMatrix{Continuous}},
    target_scitype = AbstractVector,
    load_path = "Imbalance." * string(ClusterUndersampler),
)
function MMI.transform_scitype(s::ClusterUndersampler)
    return Tuple{
        Union{Table(Continuous), AbstractMatrix{Continuous}},
        AbstractVector{<:Finite},
    }
end

"""
$(MMI.doc_header(ClusterUndersampler))

`ClusterUndersampler` implements clustering undersampling as presented in Wei-Chao, L., Chih-Fong, T., Ya-Han, H., & Jing-Shang, J. (2017). 
  Clustering-based undersampling in class-imbalanced data. Information Sciences, 409–410, 17–26. with K-means as
  the clustering algorithm.


# Training data

In MLJ or MLJBase, wrap the model in a machine by
    mach = machine(model)

There is no need to provide any data here because the model is a static transformer.

Likewise, there is no need to `fit!(mach)`. 

For default values of the hyper-parameters, model can be constructed by
    model = ClusterUndersampler()
    

# Hyperparameters

- `mode::AbstractString="nearest`: If `center` then the undersampled data will consist of the centriods of 
    each cluster found. Meanwhile, if `nearest` then it will consist of the nearest neighbor of each centroid.

$(COMMON_DOCS["RATIOS-UNDERSAMPLE"])

- `maxiter::Integer=100`: Maximum number of iterations to run K-means

- `rng::Integer=42`: Random number generator seed. Must be an integer.

# Transform Inputs

$((COMMON_DOCS["INPUTS"]))

# Transform Outputs

$(COMMON_DOCS["OUTPUTS-UNDER"])

# Operations

- `transform(mach, X, y)`: resample the data `X` and `y` using ClusterUndersampler, returning the undersampled
  versions


# Example

```
using MLJ
import Random.seed!
using MLUtils
import StatsBase.countmap

seed!(12345)

# Generate some imbalanced data:
X, y = @load_iris # a table and a vector
rand_inds = rand(1:150, 30)
X, y = getobs(X, rand_inds), y[rand_inds]

julia> countmap(y)
Dict{CategoricalArrays.CategoricalValue{String, UInt32}, Int64} with 3 entries:
  "virginica"  => 12
  "versicolor" => 5
  "setosa"     => 13

# load SMOTE model type:
ClusterUndersampler = @load ClusterUndersampler pkg=Imbalance

# Underample the majority classes to  sizes relative to the minority class:
undersampler = ClusterUndersampler(mode="nearest", ratios=Dict("setosa"=>1.0, "versicolor"=> 1.0, "virginica"=>1.0), rng=42)
mach = machine(undersampler)
X_under, y_under = transform(mach, X, y)

julia> countmap(y_under)
Dict{CategoricalArrays.CategoricalValue{String, UInt32}, Int64} with 3 entries:
  "virginica"  => 5
  "versicolor" => 5
  "setosa"     => 5
```

"""
ClusterUndersampler
