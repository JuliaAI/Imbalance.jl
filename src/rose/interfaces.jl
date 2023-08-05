### ROSE TableTransforms Interface

struct ROSE_t{T} <: TransformsBase.Transform 
    y_ind::Integer
    s::AbstractFloat
    ratios::T 
    rng::Union{Integer,AbstractRNG} 
end


"""
Instantiate a ROSE table transform

# Arguments

- `y_ind::Integer`: The index of the column containing the labels (integer-code) in the table
- `s::AbstractFloat=1.0`: A parameter that proportionally controls the bandwidth of the Gaussian kernel
$(DOC_RATIOS_ARGUMENT)
$(DOC_RNG_ARGUMENT)

# Returns

- `model::ROSE_t`: A SMOTE table transform that can be used like other transforms in TableTransforms.jl
"""
ROSE_t(y_ind::Integer;
    s::AbstractFloat=1.0,
    ratios::Union{Nothing,AbstractFloat,Dict{T,<:AbstractFloat}}=nothing, 
    rng::Union{Integer,AbstractRNG}=123) where T = ROSE_t(y_ind, s, ratios, rng)


TransformsBase.isrevertible(::Type{ROSE_t}) = true

TransformsBase.isinvertible(::Type{ROSE_t}) = false

TransformsBase.assertions(::Type{ROSE_t}) = 
  [ rose -> @assert rose.s >= 0.0 "s parameter in ROSE must be non-negative" ]


"""
Apply the ROSE transform to a table Xy

# Arguments

- `r::ROSE_t`: A ROSE table transform
- `Xy::AbstractTable`: A table where each row is an observation

# Returns

- `Xyover::AbstractTable`: A table with both the original and new observations due to ROSE
- `cache`: A cache that can be used to revert the oversampling
"""

function TransformsBase.apply(r::ROSE_t, Xy)
    for assertion in TransformsBase.assertions(ROSE_t)
        assertion(r)
    end
    Xyover = rose(Xy, r.y_ind; s = r.s, ratios = r.ratios, rng = r.rng)
    cache = rowcount(Xy)
  return Xyover, cache
end

"""
Revert the oversampling done by ROSE by removing the new observations

# Arguments

- `r::ROSE_t`: A ROSE table transform
- `Xyover::AbstractTable`: A table with both the original and new observations due to ROSE

# Returns

- `Xy::AbstractTable`: A table with only the original observations
"""
TransformsBase.revert(::ROSE_t, Xyover, cache) = revert_oversampling(Xyover, cache)

"""
Equivalent to calling `apply` but takes an extra unused argument `cache`
"""
TransformsBase.reapply(r::ROSE_t, Xy, cache) = TransformsBase.apply(r, Xy)


### ROSE MLJ Interface

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

There is no need to provide any data here because the model is a static transformer.

Likewise, there is no need to `fit!(mach)`. 

For default values of the hyper-parameters, model can be constructed by
    model = ROSE()


# Hyper-parameters

- `s::float`: A parameter that proportionally controls the bandwidth of the Gaussian kernel

$(DOC_RATIOS_ARGUMENT)

$(DOC_RNG_ARGUMENT)

# Transform Inputs

$(DOC_COMMON_INPUTS)

# Transform Outputs

$(DOC_COMMON_OUTPUTS)

# Operations

- `transform(mach, X, y)`: resample the data `X` and `y` using ROSE, returning both the
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
rose = ROSE(s=0.3, ratios=Dict("setosa"=>0.9, "versicolor"=> 1.0, "virginica"=>0.7), rng=42)
mach = machine(rose)
Xover, yover = transform(mach, X, y)

julia> countmap(yover)
Dict{CategoricalArrays.CategoricalValue{String, UInt32}, Int64} with 3 entries:
  "virginica"  => 13
  "versicolor" => 10
  "setosa"     => 13
```

"""
ROSE
