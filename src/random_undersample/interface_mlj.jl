
### RandomUndersampler with MLJ Interface
# interface struct
mutable struct RandomUndersampler{T,R<:Union{Integer,AbstractRNG}} <: Static
    ratios::T
    rng::R
    try_perserve_type::Bool
end;

"""
Initiate a random undersampling model with the given hyper-parameters.
"""
function RandomUndersampler(;
    ratios::Union{Nothing,AbstractFloat,Dict{T,<:AbstractFloat}} = 1.0,
    rng::Union{Integer,AbstractRNG} = default_rng(), try_perserve_type::Bool = true
) where {T}
    model = RandomUndersampler(ratios, rng, try_perserve_type)
    return model
end

"""
Undersample data X, y 
"""
function MMI.transform(r::RandomUndersampler, _, X, y)
    random_undersample(X, y; ratios = r.ratios, rng = r.rng, 
                      try_perserve_type = r.try_perserve_type)
end


MMI.metadata_pkg(
  RandomUndersampler,
  name = "Imbalance",
  package_uuid = "c709b415-507b-45b7-9a3d-1767c89fde68",
  package_url = "https://github.com/JuliaAI/Imbalance.jl",
  is_pure_julia = true,
)

MMI.metadata_model(
  RandomUndersampler,
  input_scitype = Union{Table(Continuous),AbstractMatrix{Continuous}},
  output_scitype = Union{Table(Continuous),AbstractMatrix{Continuous}},
  target_scitype = AbstractVector,
  load_path = "Imbalance." * string(RandomUndersampler),
)
function MMI.transform_scitype(s::RandomUndersampler)
  return Tuple{
      Union{Table(Continuous),AbstractMatrix{Continuous}},
      AbstractVector{<:Finite},
  }
end


"""
$(MMI.doc_header(RandomUndersampler))

`RandomUndersampler` implements naive undersampling by repeating existing observations
with replacement.


# Training data

In MLJ or MLJBase, wrap the model in a machine by
    mach = machine(model)

There is no need to provide any data here because the model is a static transformer.

Likewise, there is no need to `fit!(mach)`. 

For default values of the hyper-parameters, model can be constructed by
    model = RandomUndersampler()
    

# Hyperparameters

$(COMMON_DOCS["RATIOS-UNDERSAMPLE"])

$((COMMON_DOCS["RNG"]))

# Transform Inputs

$((COMMON_DOCS["INPUTS"]))

# Transform Outputs

$((COMMON_DOCS["OUTPUTS"]))

# Operations

- `transform(mach, X, y)`: resample the data `X` and `y` using RandomUndersampler, returning both the
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

# load SMOTE model type:
RandomUndersampler = @load RandomUndersampler pkg=Imbalance

# Underample the majority classes to  sizes relative to the minority class:
random_undersampler = RandomUndersampler(ratios=Dict("setosa"=>1.0, "versicolor"=> 1.0, "virginica"=>1.0), rng=42)
mach = machine(random_undersampler)
X_under, y_under = transform(mach, X, y)

julia> countmap(y_under)
Dict{CategoricalArrays.CategoricalValue{String, UInt32}, Int64} with 3 entries:
  "virginica"  => 5
  "versicolor" => 5
  "setosa"     => 5
```

"""
RandomUndersampler
