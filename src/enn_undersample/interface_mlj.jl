
### ENNUndersampler with MLJ Interface
# interface struct
mutable struct ENNUndersampler{T} <: Static
    k::Integer
    keep_condition::AbstractString
    min_ratios::T
    force_min_ratios::Bool
    rng::Integer
    try_perserve_type::Bool
end;

"""
Initiate a ENN undersampling model with the given hyper-parameters.
"""
function ENNUndersampler(;
    k::Integer = 5,
    keep_condition::AbstractString = "mode",
    min_ratios::Union{Nothing, AbstractFloat, Dict{T, <:AbstractFloat}} = 1.0,
    force_min_ratios::Bool = false,
    rng::Integer = 42,
    try_perserve_type::Bool = true,
) where {T}
    model = ENNUndersampler(
        k,
        keep_condition,
        min_ratios,
        force_min_ratios,
        rng,
        try_perserve_type,
    )
    return model
end

"""
Undersample data X, y 
"""
function MMI.transform(r::ENNUndersampler, _, X, y)
    return enn_undersample(
        X,
        y;
        k = r.k,
        keep_condition = r.keep_condition,
        min_ratios = r.min_ratios,
        force_min_ratios = r.force_min_ratios,
        rng = r.rng,
        try_perserve_type = r.try_perserve_type,
    )
end

MMI.metadata_pkg(
    ENNUndersampler,
    name = "Imbalance",
    package_uuid = "c709b415-507b-45b7-9a3d-1767c89fde68",
    package_url = "https://github.com/JuliaAI/Imbalance.jl",
    is_pure_julia = true,
)

MMI.metadata_model(
    ENNUndersampler,
    input_scitype = Union{Table(Continuous), AbstractMatrix{Continuous}},
    output_scitype = Union{Table(Continuous), AbstractMatrix{Continuous}},
    target_scitype = AbstractVector,
    load_path = "Imbalance." * string(ENNUndersampler),
)
function MMI.transform_scitype(s::ENNUndersampler)
    return Tuple{
        Union{Table(Continuous), AbstractMatrix{Continuous}},
        AbstractVector{<:Finite},
    }
end

"""
$(MMI.doc_header(ENNUndersampler))

`ENNUndersampler` undersamples a dataset by cleaning points that violate a certain condition such as
  having a different class compared to the majority of the neighbors as proposed in Dennis L Wilson. 
  Asymptotic properties of nearest neighbor rules using edited data. IEEE Transactions on Systems, Man, 
  and Cybernetics, pages 408–421, 1972.


# Training data

In MLJ or MLJBase, wrap the model in a machine by
    mach = machine(model)

There is no need to provide any data here because the model is a static transformer.

Likewise, there is no need to `fit!(mach)`. 

For default values of the hyper-parameters, model can be constructed by
    model = ENNUndersampler()
    

# Hyperparameters

$(COMMON_DOCS["K"])

- `keep_condition::AbstractString="mode"` The condition that leads to cleaning a point upon violation. Takes one of "exists", "mode", "only mode" and "all"
    - "exists": the point has at least one neighbor from the same class
    - "mode": the class of the point is one of the most frequent classes of the neighbors (there may be many)
    - "only mode": the class of the point is the single most frequent class of the neighbors
    - "all": the class of the point is the same as all the neighbors

$(COMMON_DOCS["MIN-RATIOS-UNDERSAMPLE"])

$(COMMON_DOCS["FORCE-MIN-RATIOS"])

$(COMMON_DOCS["RNG"])

$(COMMON_DOCS["TRY_PERSERVE_TYPE"])

# Transform Inputs

$((COMMON_DOCS["INPUTS"]))

# Transform Outputs

$(COMMON_DOCS["OUTPUTS-UNDER"])

# Operations

- `transform(mach, X, y)`: resample the data `X` and `y` using ENNUndersampler, returning the undersampled
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
ENNUndersampler = @load ENNUndersampler pkg=Imbalance

# Underample the majority classes to  sizes relative to the minority class:
undersampler = ENNUndersampler(keep_condition="all", min_ratios=Dict("setosa"=>1.0, "versicolor"=> 1.0, "virginica"=>1.0), rng=42)
mach = machine(undersampler)
X_under, y_under = transform(mach, X, y)

julia> countmap(y_under)
Dict{CategoricalArrays.CategoricalValue{String, UInt32}, Int64} with 3 entries:
  "virginica"  => 9
  "versicolor" => 5
  "setosa"     => 13
```

"""
ENNUndersampler
