
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
    input_scitype = Tuple{
                        Union{
                            Table(Union{Infinite, Finite}),
                            AbstractMatrix{Infinite}
                        }, 
                        AbstractVector
                    },
    output_scitype = Tuple{
        Union{
            Table(Continuous),
            AbstractMatrix{Continuous}
        }, 
        AbstractVector
    },
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
```julia
using MLJ
import Imbalance

# set probability of each class
class_probs = [0.5, 0.2, 0.3]                         
num_rows, num_continuous_feats = 100, 5
# generate a table and categorical vector accordingly
X, y = Imbalance.generate_imbalanced_data(num_rows, num_continuous_feats; 
                                class_probs, rng=42)    

julia> Imbalance.checkbalance(y)
1: ▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇ 19 (39.6%) 
2: ▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇ 33 (68.8%) 
0: ▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇ 48 (100.0%) 

# load RandomOversampler
RandomOversampler = @load RandomOversampler pkg=Imbalance

# wrap the model in a machine
oversampler = RandomOversampler(ratios=Dict(0=>1.0, 1=> 0.9, 2=>0.8), rng=42)
mach = machine(oversampler)

# provide the data to transform (there is nothing to fit)
Xover, yover = transform(mach, X, y)

julia> Imbalance.checkbalance(yover)
2: ▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇ 38 (79.2%) 
1: ▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇ 43 (89.6%) 
0: ▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇ 48 (100.0%) 
```

"""
RandomOversampler
