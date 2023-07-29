### RandomOversampler
@mlj_model mutable struct RandomOversampler <: Static
    ratios::Union{Nothing,AbstractFloat,Dict{T,<:AbstractFloat}} where {T} = nothing
    rng::Union{Integer,AbstractRNG} = default_rng()
end;

function MMI.transform(r::RandomOversampler, _, X, y)
    random_oversample(X, y; ratios = r.ratios, rng = r.rng)
end





"""
$(MMI.doc_header(RandomOversampler))

`RandomOversampler` implements naive oversampling by repeating existing observations
with replacement.


# Training data

In MLJ or MLJBase, wrap the model in a machine by
    mach = machine(model)

there is no need to provide any data here because the model is a static transformer.

Likewise, there is no need to `fit!(mach)`. 

For default values of the hyper-parameters, model can be constructed by
    model = RandomOverSampler()


# Hyper-parameters

$(DOCS_COMMON_HYPERPARAMETERS)


# Transform Inputs

$(DOCS_COMMON_INPUTS)

# Transform Outputs

$(DOCS_COMMON_OUTPUTS)

# Operations

- `transform(mach, X, y)`: resample the data `X` and `y` using SMOTE.


# Fitted parameters

There are no fitted parameters for this model.


# Example

```
using MLJBase
using Imbalance
using MLUtils
using Random
using StableRNGs: StableRNG

X, y = MLJBase.@load_iris
# Take an imbalanced subset of the data
rand_inds = rand(StableRNG(10), 1:150, 30)
X, y = getobs(X, rand_inds), y[rand_inds]
group_counts(y)
>> Dict{CategoricalArrays.CategoricalValue{String, UInt32}, Int64} with 3 entries:
  "virginica"  => 5
  "versicolor" => 15
  "setosa"     => 10

# Oversample the minority classes to  sizes relative to the majority class
R = RandomOversampler(ratios=Dict("setosa"=>0.9, "versicolor"=> 1.0, "virginica"=>0.7), rng=42)
mach = machine(R)
Xover, yover = transform(mach, X, y)
group_counts(yover)
>> Dict{CategoricalArrays.CategoricalValue{String, UInt32}, Int64} with 3 entries:
  "virginica"  => 10
  "versicolor" => 15
  "setosa"     => 14

```

"""
RandomOversampler