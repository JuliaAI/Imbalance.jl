
### TomekUndersampler with MLJ Interface
# interface struct
mutable struct TomekUndersampler{T, R <: Union{Integer, AbstractRNG}} <: Static
    min_ratios::T
    force_min_ratios::Bool
    rng::R
    try_perserve_type::Bool
end;

"""
Initiate a tomek undersampling model with the given hyper-parameters.
"""
function TomekUndersampler(;
    min_ratios::Union{Nothing, AbstractFloat, Dict{T, <:AbstractFloat}} = 1.0,
    force_min_ratios::Bool = false,
    rng::Union{Integer, AbstractRNG} = default_rng(),
    try_perserve_type::Bool = true,
) where {T}
    model = TomekUndersampler(min_ratios, force_min_ratios, rng, try_perserve_type)
    return model
end

"""
Undersample data X, y 
"""
function MMI.transform(r::TomekUndersampler, _, X, y)
    return tomek_undersample(
        X,
        y;
        min_ratios = r.min_ratios,
        force_min_ratios = r.force_min_ratios,
        rng = r.rng,
        try_perserve_type = r.try_perserve_type,
    )
end

MMI.metadata_pkg(
    TomekUndersampler,
    name = "Imbalance",
    package_uuid = "c709b415-507b-45b7-9a3d-1767c89fde68",
    package_url = "https://github.com/JuliaAI/Imbalance.jl",
    is_pure_julia = true,
)

MMI.metadata_model(
    TomekUndersampler,
    input_scitype = Union{Table(Continuous), AbstractMatrix{Continuous}},
    output_scitype = Union{Table(Continuous), AbstractMatrix{Continuous}},
    target_scitype = AbstractVector,
    load_path = "Imbalance." * string(TomekUndersampler),
)
function MMI.transform_scitype(s::TomekUndersampler)
    return Tuple{
        Union{Table(Continuous), AbstractMatrix{Continuous}},
        AbstractVector{<:Finite},
    }
end

"""
$(MMI.doc_header(TomekUndersampler))

`TomekUndersampler` undersamples by removing any point that is part of a tomek link in the data. As defined in,
 Ivan Tomek. Two modifications of cnn. IEEE Trans. Systems, Man and Cybernetics, 6:769–772, 1976.


# Training data

In MLJ or MLJBase, wrap the model in a machine by
    mach = machine(model)

There is no need to provide any data here because the model is a static transformer.

Likewise, there is no need to `fit!(mach)`. 

For default values of the hyper-parameters, model can be constructed by
    model = TomekUndersampler()
    

# Hyperparameters

$(COMMON_DOCS["MIN-RATIOS-UNDERSAMPLE"])

$(COMMON_DOCS["FORCE-MIN-RATIOS"])

$((COMMON_DOCS["RNG"]))

$(COMMON_DOCS["TRY_PERSERVE_TYPE"])

# Transform Inputs

$((COMMON_DOCS["INPUTS"]))

# Transform Outputs

$(COMMON_DOCS["OUTPUTS-UNDER"])

# Operations

- `transform(mach, X, y)`: resample the data `X` and `y` using TomekUndersampler, returning both the
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
TomekUndersampler = @load TomekUndersampler pkg=Imbalance

# Underample the majority classes to  sizes relative to the minority class:
tomek_undersampler = TomekUndersampler(min_ratios=Dict("setosa"=>1.0, "versicolor"=> 1.0, "virginica"=>1.0), rng=42)
mach = machine(tomek_undersampler)
X_under, y_under = transform(mach, X, y)

julia> countmap(y_under)
Dict{CategoricalArrays.CategoricalValue{String, UInt32}, Int64} with 3 entries:
  "virginica"  => 12
  "versicolor" => 5
  "setosa"     => 13
```

"""
TomekUndersampler
