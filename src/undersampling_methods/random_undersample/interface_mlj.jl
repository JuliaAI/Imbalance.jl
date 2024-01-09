
### RandomUndersampler with MLJ Interface
# interface struct
mutable struct RandomUndersampler{T, R <: Union{Integer, AbstractRNG}} <: Static
    ratios::T
    rng::R
    try_preserve_type::Bool
end;

"""
Initiate a random undersampling model with the given hyper-parameters.
"""
function RandomUndersampler(;
    ratios::Union{Nothing, AbstractFloat, Dict{T, <:AbstractFloat}} = 1.0,
    rng::Union{Integer, AbstractRNG} = default_rng(),
    try_preserve_type::Bool = true,
) where {T}
    model = RandomUndersampler(ratios, rng, try_preserve_type)
    return model
end

"""
Undersample data X, y 
"""
function MMI.transform(r::RandomUndersampler, _, X, y)
    return random_undersample(
        X,
        y;
        ratios = r.ratios,
        rng = r.rng,
        try_preserve_type = r.try_preserve_type,
    )
end
function MMI.transform(r::RandomUndersampler, _, X::AbstractMatrix{<:Real}, y)
    return random_undersample(
        X,
        y;
        ratios = r.ratios,
        rng = r.rng,
    )
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
    load_path = "Imbalance.MLJ.RandomUndersampler"
)
function MMI.transform_scitype(s::RandomUndersampler)
    return Tuple{
        Union{Table(Continuous), AbstractMatrix{Continuous}},
        AbstractVector{<:Finite},
    }
end

"""
$(MMI.doc_header(RandomUndersampler))

`RandomUndersampler` implements naive undersampling by randomly removing existing observations. 


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

$(COMMON_DOCS["OUTPUTS-UNDER"])

# Operations

- `transform(mach, X, y)`: resample the data `X` and `y` using RandomUndersampler, returning both the
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

julia> Imbalance.checkbalance(y; ref="minority")
 1: ▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇ 19 (100.0%) 
 2: ▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇ 33 (173.7%) 
 0: ▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇ 48 (252.6%) 

# load RandomUndersampler
RandomUndersampler = @load RandomUndersampler pkg=Imbalance

# wrap the model in a machine
undersampler = RandomUndersampler(ratios=Dict(0=>1.0, 1=> 1.0, 2=>1.0), 
               rng=42)
mach = machine(undersampler)

# provide the data to transform (there is nothing to fit)
X_under, y_under = transform(mach, X, y)
                                      
julia> Imbalance.checkbalance(y_under; ref="minority")
0: ▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇ 19 (100.0%) 
2: ▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇ 19 (100.0%) 
1: ▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇ 19 (100.0%) 
```

"""
RandomUndersampler
