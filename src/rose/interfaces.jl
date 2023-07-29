### ROSE
@mlj_model mutable struct ROSE <: Static
    s::AbstractFloat = 1.0::(_ > 0)
    ratios::Union{Nothing,AbstractFloat,Dict{T,<:AbstractFloat}} where {T} = nothing
    rng::Union{Integer,AbstractRNG} = default_rng()
end;

function MMI.transform(r::ROSE, _, X, y)
    rose(X, y; s = r.s, ratios = r.ratios, rng = r.rng)
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

there is no need to provide any data here because the model is a static transformer.

Likewise, there is no need to `fit!(mach)`. 

For default values of the hyper-parameters, model can be constructed by
    model = ROSE()


# Hyper-parameters

- `s::float`: A parameter that proportionally controls the bandwidth of the Gaussian kernel

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
R = ROSE(s=0.3, ratios=Dict("setosa"=>0.9, "versicolor"=> 1.0, "virginica"=>0.7), rng=42)
mach = machine(R)
Xover, yover = transform(mach, X, y)
group_counts(yover)
>> Dict{CategoricalArrays.CategoricalValue{String, UInt32}, Int64} with 3 entries:
  "virginica"  => 10
  "versicolor" => 15
  "setosa"     => 14

```

"""
ROSE
