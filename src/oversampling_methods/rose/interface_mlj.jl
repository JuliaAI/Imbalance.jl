### ROSE MLJ Interface
# interface struct
mutable struct ROSE{T,R<:Union{Integer,AbstractRNG}, F<:AbstractFloat} <: MMI.Static
    s::F
    ratios::T
    rng::R
    try_perserve_type::Bool
end;



"""
Check whether the given model hyperparameters are valid and clean them if necessary. 
"""
function MMI.clean!(r::ROSE)
    message = ""
    if r.s < 0
        throw((ERR_NONNEG_S(r.s)))
    end
    return message
end

"""
Initiate a ROSE model with the given hyper-parameters.
"""
function ROSE(;
    s::AbstractFloat = 1.0,
    ratios::Union{Nothing,AbstractFloat,Dict{T,<:AbstractFloat}} = 1.0,
    rng::Union{Integer,AbstractRNG} = default_rng(), try_perserve_type::Bool = true,
) where {T}
    model = ROSE(s, ratios, rng, try_perserve_type)
    MMI.clean!(model)
    return model
end

"""
Oversample data X, y using ROSE
"""
function MMI.transform(r::ROSE, _, X, y)
    rose(X, y; s = r.s, ratios = r.ratios, rng = r.rng, 
        try_perserve_type = r.try_perserve_type)
end
function MMI.transform(r::ROSE, _, X::AbstractMatrix{<:Real}, y)
    rose(X, y; s = r.s, ratios = r.ratios, rng = r.rng,)
end

MMI.metadata_pkg(
    ROSE,
    name = "Imbalance",
    package_uuid = "c709b415-507b-45b7-9a3d-1767c89fde68",
    package_url = "https://github.com/JuliaAI/Imbalance.jl",
    is_pure_julia = true,
)

MMI.metadata_model(
    ROSE,
    input_scitype = Union{Table(Continuous),AbstractMatrix{Continuous}},
    output_scitype = Union{Table(Continuous),AbstractMatrix{Continuous}},
    target_scitype = AbstractVector,
    load_path = "Imbalance.MLJ.ROSE"
)
function MMI.transform_scitype(s::ROSE)
    return Tuple{
        Union{Table(Continuous),AbstractMatrix{Continuous}},
        AbstractVector{<:Finite},
    }
end


"""
$(MMI.doc_header(ROSE))

`ROSE` implements the ROSE (Random Oversampling Examples) algorithm to 
    correct for class imbalance as in G Menardi, N. Torelli, “Training and assessing 
    classification rules with imbalanced data,” 
    Data Mining and Knowledge Discovery, 28(1), pp.92-122, 2014.


# Training data

In MLJ or MLJBase, wrap the model in a machine by
    mach = machine(model)

There is no need to provide any data here because the model is a static transformer.

Likewise, there is no need to `fit!(mach)`. 

For default values of the hyper-parameters, model can be constructed by
    model = ROSE()


# Hyperparameters

- `s::float`: A parameter that proportionally controls the bandwidth of the Gaussian kernel

$((COMMON_DOCS["RATIOS"]))

$((COMMON_DOCS["RNG"]))

# Transform Inputs

$((COMMON_DOCS["INPUTS"]))

# Transform Outputs

$((COMMON_DOCS["OUTPUTS"]))

# Operations

- `transform(mach, X, y)`: resample the data `X` and `y` using ROSE, returning both the
  new and original observations

# Example

```julia
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

# load ROSE model type:
ROSE = @load ROSE pkg=Imbalance

# Oversample the minority classes to  sizes relative to the majority class:
rose = ROSE(s=0.3, ratios=Dict("setosa"=>0.9, "versicolor"=> 1.0, "virginica"=>0.7), rng=42)
mach = machine(rose)
Xover, yover = transform(mach, X, y)

julia> countmap(yover)
Dict{CategoricalArrays.CategoricalValue{String, UInt32}, Int64} with 3 entries:
  "virginica"  => 13
  "versicolor" => 10
  "setosa"     => 13
```

"""
ROSE
