
### RandomOversampler with MLJ Interface
# interface struct
mutable struct RandomOversampler{T,R<:Union{Integer,AbstractRNG}} <: Static
    ratios::T
    rng::R
    try_preserve_type::Bool
end;

"""
Initiate a random oversampling model with the given hyper-parameters.
"""
function RandomOversampler(;
    ratios::Union{Nothing,AbstractFloat,Dict{T,<:AbstractFloat}} = 1.0,
    rng::Union{Integer,AbstractRNG} = default_rng(), try_preserve_type::Bool = true
) where {T}
    model = RandomOversampler(ratios, rng, try_preserve_type)
    return model
end

"""
Oversample data X, y using ROSE
"""
function MMI.transform(r::RandomOversampler, _, X, y)
    random_oversample(X, y; ratios = r.ratios, rng = r.rng, 
                      try_preserve_type = r.try_preserve_type)
end
function MMI.transform(r::RandomOversampler, _, X::AbstractMatrix{<:Real}, y)
  random_oversample(X, y; ratios = r.ratios, rng = r.rng)
end

MMI.metadata_pkg(
  RandomOversampler,
  name = "Imbalance",
  package_uuid = "c709b415-507b-45b7-9a3d-1767c89fde68",
  package_url = "https://github.com/JuliaAI/Imbalance.jl",
  is_pure_julia = true,
)

MMI.metadata_model(
  RandomOversampler,
  input_scitype = Union{Table(Continuous),AbstractMatrix{Continuous}},
  output_scitype = Union{Table(Continuous),AbstractMatrix{Continuous}},
  target_scitype = AbstractVector,
  load_path = "Imbalance.MLJ.RandomOversampler" 
)
function MMI.transform_scitype(s::RandomOversampler)
  return Tuple{
      Union{Table(Continuous),AbstractMatrix{Continuous}},
      AbstractVector{<:Finite},
  }
end


"""
$(MMI.doc_header(RandomOversampler))

`RandomOversampler` implements naive oversampling by repeating existing observations
with replacement.


# Training data

In MLJ or MLJBase, wrap the model in a machine by
    mach = machine(model)

There is no need to provide any data here because the model is a static transformer.

Likewise, there is no need to `fit!(mach)`. 

For default values of the hyper-parameters, model can be constructed by
    model = RandomOverSampler()
    

# Hyperparameters

$((COMMON_DOCS["RATIOS"]))

$((COMMON_DOCS["RNG"]))

# Transform Inputs

$((COMMON_DOCS["INPUTS"]))

# Transform Outputs

$((COMMON_DOCS["OUTPUTS"]))

# Operations

- `transform(mach, X, y)`: resample the data `X` and `y` using RandomOversampler, returning both the
  new and original observations


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

# load RandomOversampler model type:
RandomOversampler = @load RandomOversampler pkg=Imbalance

# Oversample the minority classes to  sizes relative to the majority class:
random_oversampler = RandomOversampler(ratios=Dict("setosa"=>0.9, "versicolor"=> 1.0, "virginica"=>0.7), rng=42)
mach = machine(random_oversampler)
Xover, yover = transform(mach, X, y)

julia> countmap(yover)
Dict{CategoricalArrays.CategoricalValue{String, UInt32}, Int64} with 3 entries:
  "virginica"  => 13
  "versicolor" => 10
  "setosa"     => 13
```

"""
RandomOversampler
