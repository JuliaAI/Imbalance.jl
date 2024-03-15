"""
Assuming that all the observations in the observation matrix X belong to the same class, 
randomly remove n observations for that class using random undersampling

# Arguments
- `X`: A matrix where each column is an observation of floats
- `n`: Number of observations to keep

# Returns
- `Xnew`: A matrix containing the undersampled observations
"""
function random_undersample_per_class(
    X::AbstractMatrix{<:Union{Real, Missing}},
    n::Integer;
    rng::AbstractRNG = default_rng(),
)
    # sample n rows from X by sampling indices
    random_inds = sample(rng, 1:size(X, 2), n; replace = false, ordered = true)
    Xnew = X[:, random_inds]
    return Xnew
end

"""
    random_undersample(
        X, y; 
        ratios=1.0, rng=default_rng(), 
        try_preserve_type=true
    )


# Description

Naively undersample a dataset by randomly deleting existing observations.

# Positional Arguments

- `X`: A matrix of real numbers or a table with element [scitypes](https://juliaai.github.io/ScientificTypes.jl/) that subtype `Union{Finite, Infinite}`. 
     Elements in nominal columns should subtype `Finite` (i.e., have [scitype](https://juliaai.github.io/ScientificTypes.jl/) `OrderedFactor` or `Multiclass`) and
     elements in continuous columns should subtype `Infinite` (i.e., have [scitype](https://juliaai.github.io/ScientificTypes.jl/) `Count` or `Continuous`).

- `y`: An abstract vector of labels (e.g., strings) that correspond to the observations in `X`

# Keyword Arguments

$(COMMON_DOCS["RATIOS-UNDERSAMPLE"])

$(COMMON_DOCS["RNG"])

$(COMMON_DOCS["TRY_PRESERVE_TYPE"])

# Returns

$(COMMON_DOCS["OUTPUTS-UNDER"])


# Example
```julia
using Imbalance

# set probability of each class
class_probs = [0.5, 0.2, 0.3]                         
num_rows, num_continuous_feats = 100, 5
# generate a table and categorical vector accordingly
X, y = generate_imbalanced_data(num_rows, num_continuous_feats; 
                                class_probs, rng=42)   

julia> Imbalance.checkbalance(y; ref="minority")
 1: ▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇ 19 (100.0%) 
 2: ▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇ 33 (173.7%) 
 0: ▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇ 48 (252.6%) 

# apply randomundersampling
X_under, y_under = random_undersample(X, y; ratios=Dict(0=>1.0, 1=> 1.0, 2=>1.0), 
                                      rng=42)
                                      
julia> Imbalance.checkbalance(y_under; ref="minority")
0: ▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇ 19 (100.0%) 
2: ▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇ 19 (100.0%) 
1: ▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇ 19 (100.0%) 
```

# MLJ Model Interface

Simply pass the keyword arguments while initiating the `RandomUndersampler` model and pass the 
    positional arguments `X, y` to the `transform` method. 

```julia
using MLJ
RandomUndersampler = @load RandomUndersampler pkg=Imbalance

# Wrap the model in a machine
undersampler = RandomUndersampler(ratios=Dict(0=>1.0, 1=> 1.0, 2=>1.0), 
               rng=42)
mach = machine(undersampler)

# Provide the data to transform (there is nothing to fit)
X_under, y_under = transform(mach, X, y)
```
You can read more about this `MLJ` interface by accessing it from MLJ's [model browser](https://alan-turing-institute.github.io/MLJ.jl/dev/model_browser/).


# TableTransforms Interface

This interface assumes that the input is one table `Xy` and that `y` is one of the columns. Hence, an integer `y_ind`
    must be specified to the constructor to specify which column `y` is followed by other keyword arguments. 
    Only `Xy` is provided while applying the transform.

```julia
using Imbalance
using Imbalance.TableTransforms

# Generate imbalanced data
num_rows = 100
num_features = 5
y_ind = 3
Xy, _ = generate_imbalanced_data(num_rows, num_features; 
                                 class_probs=[0.5, 0.2, 0.3], insert_y=y_ind, rng=42)

# Initiate Random Undersampler model
undersampler = RandomUndersampler(y_ind; ratios=Dict(0=>1.0, 1=>1.0, 2=>1.0), rng=42)
Xy_under = Xy |> undersampler                    
Xy_under, cache = TableTransforms.apply(undersampler, Xy)    # equivalently
```
The `reapply(undersampler, Xy, cache)` method from `TableTransforms` simply falls back to `apply(undersample, Xy)` and the `revert(undersampler, Xy, cache)`
is not supported.

# Illustration
A full basic example along with an animation can be found [here](https://githubtocolab.com/JuliaAI/Imbalance.jl/blob/dev/examples/undersample_random.ipynb). 
    You may find more practical examples in the [tutorial](https://juliaai.github.io/Imbalance.jl/dev/examples/) 
    section which also explains running code on Google Colab.
"""
function random_undersample(
    X::AbstractMatrix{<:Union{Real, Missing}},
    y::AbstractVector;
    ratios = 1.0,
    rng::Union{AbstractRNG, Integer} = default_rng(),
    try_preserve_type::Bool = true,
)
    rng = rng_handler(rng)
    X_under, y_under = generic_undersample(X, y, random_undersample_per_class; ratios, rng)
    return X_under, y_under
end

# dispatch for when X is a table
function random_undersample(
    X,
    y::AbstractVector;
    ratios = 1.0,
    rng::Union{AbstractRNG, Integer} = default_rng(),
    try_preserve_type::Bool = true,
)
    X_under, y_under = tablify(
        random_undersample,
        X,
        y;
        try_preserve_type = try_preserve_type,
        encode_func = generic_encoder,
        decode_func = generic_decoder,
        ratios,
        rng,
    )
    return X_under, y_under
end

# dispatch for table inputs where y is one of the columns
function random_undersample(
    Xy,
    y_ind::Integer;
    ratios = 1.0,
    rng::Union{AbstractRNG, Integer} = default_rng(),
    try_preserve_type::Bool = true,
)
    Xy_under = tablify(
        random_undersample,
        Xy,
        y_ind;
        try_preserve_type = try_preserve_type,
        encode_func = generic_encoder,
        decode_func = generic_decoder,
        ratios,
        rng,
    )
    return Xy_under
end
