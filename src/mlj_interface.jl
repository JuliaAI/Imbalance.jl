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


### ROSE
@mlj_model mutable struct ROSE <: Static
    s::AbstractFloat = 1.0::(_ > 0)
    ratios::Union{Nothing,AbstractFloat,Dict{T,<:AbstractFloat}} where {T} = nothing
    rng::Union{Integer,AbstractRNG} = default_rng()
end;

function MMI.transform(r::ROSE, _, X, y)
    rose(X, y; s = r.s, ratios = r.ratios, rng = r.rng)
end


### RandomOversampler
@mlj_model mutable struct RandomOversampler <: Static
    ratios::Union{Nothing,AbstractFloat,Dict{T,<:AbstractFloat}} where {T} = nothing
    rng::Union{Integer,AbstractRNG} = default_rng()
end;

function MMI.transform(r::RandomOversampler, _, X, y)
    random_oversample(X, y; ratios = r.ratios, rng = r.rng)
end


for model_name in [:SMOTE, :ROSE, :RandomOversampler]
    quote
        MMI.metadata_pkg(
            $model_name,
            name = "Imbalance",
            package_uuid = "c709b415-507b-45b7-9a3d-1767c89fde68",
            package_url = "https://github.com/EssamWisam/Imbalance.jl",
            is_pure_julia = true,
        )

        MMI.metadata_model(
            $model_name,
            input_scitype = Union{Table(Continuous),AbstractMatrix{Continuous}},
            output_scitype = Union{Table(Continuous),AbstractMatrix{Continuous}},
            load_path = "Imbalance."* string($model_name),
        )
        print("Imbalance."* string($model_name))
        function MMI.transform_scitype(s::$model_name)
            return Tuple{
                Union{Table(Continuous),AbstractMatrix{Continuous}},
                AbstractVector{<:Finite},
            }
        end
    end |> eval
end 

const DOCS_COMMON_HYPERPARAMETERS = 
"""
- `ratios=nothing`: A parameter that controls the amount of oversampling to be done for each class.
    - Can be a dictionary mapping each class to the ratio of the needed number of observations
     for that class to the initial number of observations of the majority class.
    - Can be nothing and in this case each class will be oversampled to the size of 
    the majority class.
    - Can be a float and in this case each class will be oversampled 
    to the size of the majority class times the float.
    
- `rng=Random.default_rng()`: Either an `AbstractRNG` object or an `Integer` 
    seed to be used with `StableRNG`.
"""

const DOCS_COMMON_INPUTS = 
"""
- `X`: A matrix or table where each row is an observation (vector) of floats

- `y`: An abstract vector of labels that correspond to the observations in `X`
"""

const DOCS_COMMON_OUTPUTS = 
"""
- `Xover`: A matrix or table like X (if possible, else a columntable) depending on whether X 
    is a matrix or table respectively that includes original data and the new observations 
    due to oversampling.

- `yover`: An abstract vector of labels that includes the original
    labels and the new instances of them due to oversampling.
"""

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