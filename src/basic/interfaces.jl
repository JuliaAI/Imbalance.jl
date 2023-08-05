### Random Oversample TableTransforms Interface

struct RandomOversampler_t{T} <: Transform
    y_ind::Integer
    ratios::T
    rng::Union{Integer,AbstractRNG}
end

"""
Instantiate a naive RandomOversampler table transform

# Arguments

- `y_ind::Integer`: The index of the column containing the labels (integer-code) in the table
$(DOC_RATIOS_ARGUMENT)
$(DOC_RNG_ARGUMENT)

# Returns

- `model::RandomOversampler_t`: A SMOTE table transform that can be used like other transforms in TableTransforms.jl
"""
RandomOversampler_t(
    y_ind::Integer;
    ratios::Union{Nothing,AbstractFloat,Dict{T,<:AbstractFloat}} = nothing,
    rng::Union{Integer,AbstractRNG} = 123,
) where {T} = RandomOversampler_t(y_ind, ratios, rng)


TransformsBase.isrevertible(::Type{RandomOversampler_t}) = true

TransformsBase.isinvertible(::Type{RandomOversampler_t}) = false

"""
Apply the RandomOversampler transform to a table Xy

# Arguments

- `r::RandomOversampler_t`: A RandomOversampler table transform
- `Xy::AbstractTable`: A table where each row is an observation

# Returns

- `Xyover::AbstractTable`: A table with both the original and new observations due to RandomOversampler
- `cache`: A cache that can be used to revert the oversampling
"""
function TransformsBase.apply(r::RandomOversampler_t, Xy)
    Xyover = random_oversample(Xy, r.y_ind; ratios = r.ratios, rng = r.rng)
    cache = rowcount(Xy)
    return Xyover, cache
end

"""
Revert the oversampling done by RandomOversampler by removing the new observations

# Arguments

- `r::RandomOversampler_t`: A RandomOversampler table transform
- `Xyover::AbstractTable`: A table with both the original and new observations due to RandomOversampler

# Returns

- `Xy::AbstractTable`: A table with only the original observations
"""
TransformsBase.revert(::RandomOversampler_t, Xyover, cache) =
    revert_oversampling(Xyover, cache)

"""
Equivalent to `apply(r, Xy)`
"""
TransformsBase.reapply(r::RandomOversampler_t, Xy, cache) = TransformsBase.apply(r, Xy)



### RandomOversampler with MLJ Interface
mutable struct RandomOversampler{T} <: Static
    ratios::T
    rng::Union{Integer,AbstractRNG}
end;


function RandomOversampler(;ratios::Union{Nothing,AbstractFloat,Dict{T,<:AbstractFloat}} = nothing, 
  rng::Union{Integer,AbstractRNG} = default_rng()
) where {T}
model = RandomOversampler(ratios, rng)
return model
end

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

There is no need to provide any data here because the model is a static transformer.

Likewise, there is no need to `fit!(mach)`. 

For default values of the hyper-parameters, model can be constructed by
    model = RandomOverSampler()
    

# Hyper-parameters

$(DOC_RATIOS_ARGUMENT)

$(DOC_RNG_ARGUMENT)

# Transform Inputs

$(DOC_COMMON_INPUTS)

# Transform Outputs

$(DOC_COMMON_OUTPUTS)

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

# load SMOTE model type:
SMOTE = @load SMOTE pkg=Imbalance

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
