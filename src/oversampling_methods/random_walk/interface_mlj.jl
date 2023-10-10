### RandomWalkOversampler with MLJ Interface

mutable struct RandomWalkOversampler{T,R<:Union{Integer,AbstractRNG}} <: Static
    ratios::T
    rng::R
    try_preserve_type::Bool
end




"""
Initiate a RandomWalkOversampler model with the given hyper-parameters.
"""
function RandomWalkOversampler(;
    ratios::Union{Nothing,AbstractFloat,Dict{T,<:AbstractFloat}} =1.0,
    rng::Union{Integer,AbstractRNG} = default_rng(),
    try_preserve_type::Bool=true
) where {T}
    model = RandomWalkOversampler(ratios, rng, try_preserve_type)
    return model
end




"""
Oversample data X, y using RandomWalkOversampler
"""
function MMI.transform(s::RandomWalkOversampler, _, X, y)
    random_walk_oversample(X, y; ratios = s.ratios, rng = s.rng, try_preserve_type=s.try_preserve_type)
end




MMI.metadata_pkg(
    RandomWalkOversampler,
    name = "Imbalance",
    package_uuid = "c709b415-507b-45b7-9a3d-1767c89fde68",
    package_url = "https://github.com/JuliaAI/Imbalance.jl",
    is_pure_julia = true,
)

MMI.metadata_model(
    RandomWalkOversampler,
    input_scitype = Union{
        Table(Union{Infinite, Finite}),
    },
    output_scitype = Union{
        Table(Union{Infinite, Finite}),
    },
    target_scitype = AbstractVector,
    load_path = "Imbalance.MLJ.RandomWalkOversampler"
)


function MMI.transform_scitype(s::RandomWalkOversampler)
    return Tuple{
        Union{
            Table(Union{Infinite,OrderedFactor,Multiclass}),
        },
        AbstractVector{<:Finite},
    }
end



"""
$(MMI.doc_header(RandomWalkOversampler))

`RandomWalkOversampler` implements the random walk oversampling algorithm to correct for class imbalance as in
    Zhang, H., & Li, M. (2014). RWO-Sampling: A random walk over-sampling approach to imbalanced data classification. 
    Information Fusion, 25, 4-20.

# Training data

In MLJ or MLJBase, wrap the model in a machine by

    mach = machine(model)

There is no need to provide any data here because the model is a static transformer.

Likewise, there is no need to `fit!(mach)`.

For default values of the hyper-parameters, model can be constructed by

    model = RandomWalkOversampler()


# Hyperparameters

$((COMMON_DOCS["RATIOS"]))

$((COMMON_DOCS["RNG"]))

# Transform Inputs

- `X`: A matrix of floats or a table with element [scitypes](https://juliaai.github.io/ScientificTypes.jl/) 
     that subtype `Union{Finite, Infinite}`. Elements in nominal columns should subtype `Finite` 
     (i.e., have [scitype](https://juliaai.github.io/ScientificTypes.jl/) `OrderedFactor` or `Multiclass`) and
	 elements in continuous columns should subtype `Infinite` (i.e., have 
     [scitype](https://juliaai.github.io/ScientificTypes.jl/) `Count` or `Continuous`).

- `y`: An abstract vector of labels (e.g., strings) that correspond to the observations in `X`

# Transform Outputs

$((COMMON_DOCS["OUTPUTS"]))

# Operations

- `transform(mach, X, y)`: resample the data `X` and `y` using RandomWalkOversampler, returning both the
  new and original observations

# Example

```
using MLJ
import Random.seed!
import Imbalance

# set probability of each class
class_probs = [0.5, 0.2, 0.3]                         
num_rows = 100
num_continuous_feats = 3
# want two categorical features with three and two possible values respectively
num_vals_per_category = [3, 2]

# generate a table and categorical vector accordingly
X, y = Imbalance.generate_imbalanced_data(num_rows, num_continuous_feats; 
								class_probs, num_vals_per_category, rng=42)                      
julia> Imbalance.checkbalance(y)
1: ▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇ 19 (39.6%) 
2: ▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇ 33 (68.8%) 
0: ▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇ 48 (100.0%) 


julia> ScientificTypes.schema(X).scitypes
(Continuous, Continuous, Continuous, Continuous, Continuous)
# coerce nominal columns to a finite scitype (multiclass or ordered factor)
X = coerce(X, :Column4=>Multiclass, :Column5=>Multiclass)

# load RandomWalkOversampler model type:
RandomWalkOversampler = @load RandomWalkOversampler pkg=Imbalance

# oversample the minority classes to  sizes relative to the majority class:
oversampler = RandomWalkOversampler(ratios = Dict(0=>1.0, 1=> 0.9, 2=>0.8), rng = 42)
mach = machine(oversampler)
Xover, yover = transform(mach, X, y)

julia> Imbalance.checkbalance(yover)
2: ▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇ 38 (79.2%) 
1: ▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇ 43 (89.6%) 
0: ▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇ 48 (100.0%)
```
"""
RandomWalkOversampler

