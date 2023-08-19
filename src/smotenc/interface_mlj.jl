### SMOTENC with MLJ Interface

mutable struct SMOTENC{T} <: Static
    k::Integer 
    ratios::T
    rng::Union{Integer,AbstractRNG}
end;

"""
Check whether the given model hyperparameters are valid and clean them if necessary. 
"""
function MMI.clean!(s::SMOTENC)
  message = ""
    if s.k < 1
        message = "k for SMOTENC must be at least 1 but found $(s.k). Setting k = 1."
        s.k = 1
    end
    return message
end

"""
Initiate a SMOTENC model with the given hyper-parameters.
"""
function SMOTENC(; k::Integer = 5, 
        ratios::Union{Nothing,AbstractFloat,Dict{T,<:AbstractFloat}} = nothing, 
        rng::Union{Integer,AbstractRNG} = default_rng()
) where {T}
    model = SMOTENC(k, ratios, rng)
    message = MMI.clean!(model)
    isempty(message) || @warn message
    return model
end

"""
Oversample data X, y using SMOTENC
"""
function MMI.transform(s::SMOTENC, _, X, y)
    smotenc(X, y; k = s.k, ratios = s.ratios, rng = s.rng)
end



"""
$(MMI.doc_header(SMOTENC))

`SMOTENC` implements the SMOTENC algorithm to correct for class imbalance as in
N. V. Chawla, K. W. Bowyer, L. O.Hall, W. P. Kegelmeyer,
“SMOTE: synthetic minority over-sampling technique,”
Journal of artificial intelligence research, 321-357, 2002.


# Training data

In MLJ or MLJBase, wrap the model in a machine by

    mach = machine(model)

There is no need to provide any data here because the model is a static transformer.

Likewise, there is no need to `fit!(mach)`.

For default values of the hyper-parameters, model can be constructed by

    model = SMOTENC()


# Hyper-parameters

- `k=5`: Number of nearest neighbors to consider in the SMOTENC algorithm.  Should be within
    the range `[1, n - 1]`, where `n` is the number of observations; otherwise set to the
    nearest of these two values.

$(DOC_RATIOS_ARGUMENT)

$(DOC_RNG_ARGUMENT)

# Transform Inputs

$(DOC_COMMON_INPUTS)

# Transform Outputs

$(DOC_COMMON_OUTPUTS)

# Operations

- `transform(mach, X, y)`: resample the data `X` and `y` using SMOTENC, returning both the
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

# load SMOTENC model type:
SMOTENC = @load SMOTENC pkg=Imbalance

# Oversample the minority classes to  sizes relative to the majority class:
smote = SMOTENC(k=10, ratios=Dict("setosa"=>1.0, "versicolor"=> 0.8, "virginica"=>1.0), rng=42)
mach = machine(smote)
Xover, yover = transform(mach, X, y)

julia> countmap(yover)
Dict{CategoricalArrays.CategoricalValue{String, UInt32}, Int64} with 3 entries:
  "virginica"  => 13
  "versicolor" => 10
  "setosa"     => 13
```

"""
SMOTENC
