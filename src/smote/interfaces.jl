### SMOTE
@mlj_model mutable struct SMOTE <: Static
    # TODO: add check for k > 0 and others
    k::Integer = 5::(_ > 0)
    ratios::Union{Nothing,AbstractFloat,Dict{T,<:AbstractFloat}} where {T} = nothing
    rng::Union{Integer,AbstractRNG} = default_rng()
end;

function MMI.transform(s::SMOTE, _, X, y)
    smote(X, y; k = s.k, ratios = s.ratios, rng = s.rng)
end



"""
$(MMI.doc_header(SMOTE))

`SMOTE` implements the SMOTE algorithm to correct for class imbalance as in
N. V. Chawla, K. W. Bowyer, L. O.Hall, W. P. Kegelmeyer,
“SMOTE: synthetic minority over-sampling technique,”
Journal of artificial intelligence research, 321-357, 2002.


# Training data

In MLJ or MLJBase, wrap the model in a machine by
    mach = machine(model)

there is no need to provide any data here because the model is a static transformer.

Likewise, there is no need to `fit!(mach)`.

For default values of the hyper-parameters, model can be constructed by
    model = SMOTE()


# Hyper-parameters

- `k=5`: Number of nearest neighbors to consider in the SMOTE algorithm.
    Should be within the range `[1, size(X, 1) - 1]` else set to the nearest of these two values.

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
S = SMOTE(k=10, ratios=Dict("setosa"=>0.9, "versicolor"=> 1.0, "virginica"=>0.7), rng=42)
mach = machine(S)
Xover, yover = transform(mach, X, y)
group_counts(yover)
>> Dict{CategoricalArrays.CategoricalValue{String, UInt32}, Int64} with 3 entries:
  "virginica"  => 10
  "versicolor" => 15
  "setosa"     => 14

```

"""
SMOTE
