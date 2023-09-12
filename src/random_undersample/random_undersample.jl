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
    X::AbstractMatrix{<:Real},
    n::Integer;
    rng::AbstractRNG = default_rng(),
)
    # sample n rows from X by sampling indices
    random_inds = sample(rng, 1:size(X, 2), n; replace = false, ordered=true)
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

$(COMMON_DOCS["INPUTS"])

# Keyword Arguments

$(COMMON_DOCS["RATIOS-UNDERSAMPLE"])

$(COMMON_DOCS["RNG"])

$(COMMON_DOCS["TRY_PERSERVE_TYPE"])

# Returns

$(COMMON_DOCS["OUTPUTS"])


# Example
```julia
using Imbalance

# set probability of each class
probs = [0.5, 0.2, 0.3]                         
num_rows, num_continuous_feats = 100, 5
# generate a table and categorical vector accordingly
X, y = generate_imbalanced_data(num_rows, num_continuous_feats; 
                                probs, rng=42)                       
julia> checkbalance(y; ref="minority")
 1: ▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇ 19 (100.0%) 
 2: ▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇ 33 (173.7%) 
 0: ▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇ 48 (252.6%) 

# apply randomundersampling
X_under, y_under = random_undersample(X, y; ratios=Dict(0=>1.0, 1=> 1.0, 2=>1.0), rng=42)
checkbalance(y_under)
0: ▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇ 19 (100.0%) 
2: ▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇ 19 (100.0%) 
1: ▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇ 19 (100.0%) 
```

# MLJ Model Interface

Simply pass the keyword arguments while initiating the `RandomUndersampler` model and pass the 
    positional arguments to the `transform` method. 

```julia
using MLJ
RandomUndersampler = @load RandomUndersampler pkg=Imbalance

# Wrap the model in a machine
undersampler = RandomUndersampler(ratios=Dict(0=>1.0, 1=> 1.0, 2=>1.0), rng=42)
mach = machine(undersampler)

# Provide the data to transform (there is nothing to fit)
X_under, y_under = transform(mach, X, y)
```
The `MLJ` interface is only supported for table inputs. Read more about the interface [here]().

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
                                 probs=[0.5, 0.2, 0.3], insert_y=y_ind, rng=42)

# Initiate Random Undersampler model
undersampler = RandomUndersampler(y_ind; ratios=Dict(0=>1.0, 1=> 0.9, 2=>0.8), rng=42)
Xy_under = Xy |> undersampler                    
Xy_under, cache = TableTransforms.apply(undersampler, Xy)    # equivalently
```
The `reapply(undersampler, Xy, cache)` method from `TableTransforms` simply falls back to `apply(undersample, Xy)` and the `revert(undersampler, Xy, cache)`
is not supported.
"""
function random_undersample(
    X::AbstractMatrix{<:Real},
    y::AbstractVector;
    ratios = 1.0,
    rng::Union{AbstractRNG,Integer} = default_rng(),
)
    rng = rng_handler(rng)
    X_under, y_under = generic_undersample(X, y, random_undersample_per_class; ratios, rng,)
    return X_under, y_under
end

# dispatch for when X is a table
function random_undersample(
    X,
    y::AbstractVector;
    ratios = 1.0,
    rng::Union{AbstractRNG,Integer} = default_rng(),
    try_perserve_type::Bool=true
)
    X_under, y_under = tablify(random_undersample, X, y; 
                           try_perserve_type=try_perserve_type, 
                           encode_func = generic_encoder,
                           decode_func = generic_decoder,
                           ratios, 
                           rng)
    return X_under, y_under
end


# dispatch for table inputs where y is one of the columns
function random_undersample(
    Xy,
    y_ind::Integer;
    ratios = 1.0,
    rng::Union{AbstractRNG,Integer} = default_rng(),
    try_perserve_type::Bool=true
)
    Xy_under = tablify(random_undersample, Xy, y_ind; 
                    try_perserve_type=try_perserve_type, 
                    encode_func = generic_encoder,
                    decode_func = generic_decoder,
                    ratios, rng)
    return Xy_under
end
