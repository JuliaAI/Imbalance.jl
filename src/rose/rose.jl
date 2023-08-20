"""
Assuming that all the observations in the observation matrix X belong to the same class, 
generate n new observations for that class using ROSE.

# Arguments
- `X::AbstractMatrix`: A matrix where each row is an observation of floats
- `n::Int`: Number of new observations to generate
- `s::float`: A parameter that proportionally controls the bandwidth of the Gaussian kernel
- `rng::AbstractRNG`: Random number generator

# Returns
- `AbstractMatrix`: A matrix where each row is a new observation generated by ROSE
"""
function rose_per_class(
    X::AbstractMatrix{<:AbstractFloat},
    n::Int;
    s::AbstractFloat = 1.0,
    rng::AbstractRNG = default_rng(),
)
    # sample n rows from X
    Xnew = randcols(rng, X, n)
    # For s == 0 this is just random oversampling
    if s == 0.0
        return Xnew
    end

    # compute the standard deviation feature-wise
    σs = vec(std(Xnew, dims = 2))
    # compute h and then H as in the paper
    d = size(Xnew, 1)
    N = size(Xnew, 2)
    h = (4 / ((d + 2) * N))^(1 / (d + 4))
    # make a diagonal matrix of the result
    H = Diagonal(σs * s * h)

    # generate standard normal samples of same dimension of Xnew
    XSnew = randn(rng, size(Xnew))
    # matrix multiply the diagonal matrix by XSnew
    XSnew = H * XSnew
    # add Xnew and XSnew
    Xnew += XSnew
    # This is equivalent to sampling from a multivariate normal
    # centered at each point in Xnew with covariance matrix H

    # return the result
    return Xnew
end

"""
    rose(
        X, y; 
        s::AbstractFloat=0.1, ratios=nothing, rng::AbstractRNG=default_rng(),
        try_perserve_type=true
    )

# Description

Oversamples a dataset using `ROSE` (Random Oversampling Examples) algorithm to 
    correct for class imbalance as presented in [1]


# Positional Arguments


$DOC_COMMON_INPUTS

# Keyword Arguments

- `s::float`: A parameter that proportionally controls the bandwidth of the Gaussian kernel

$DOC_RATIOS_ARGUMENT

$DOC_RNG_ARGUMENT

$DOC_TRY_PERSERVE_ARGUMENT


# Returns

$DOC_COMMON_OUTPUTS

# Example

```julia
using Imbalance
using StatsBase

# set probability of each class
probs = [0.5, 0.2, 0.3]                         
num_rows, num_cont_feats = 100, 5
# generate a table and categorical vector accordingly
X, y = generate_imbalanced_data(num_rows, num_cont_feats; 
                                probs, rng=42)                       
StatsBase.countmap(y)

julia> Dict{CategoricalArrays.CategoricalValue{Int64, UInt32}, Int64} with 3 entries:
0 => 48
2 => 33
1 => 19

# apply ROSE
Xover, yover = rose(X, y; s=0.3, ratios=Dict(0=>1.0, 1=> 0.9, 2=>0.8), rng=42)
StatsBase.countmap(yover)

julia> Dict{CategoricalArrays.CategoricalValue{Int64, UInt32}, Int64} with 3 entries:
0 => 48
2 => 38
1 => 43
```

# MLJ Model Interface

Simply pass the keyword arguments while initiating the `ROSE` model and pass the 
    positional arguments to the `transform` method. 

```julia
using MLJ
ROSE = @load ROSE pkg=Imbalance

# Wrap the model in a machine
oversampler = ROSE(s=0.3, ratios=Dict(0=>1.0, 1=> 0.9, 2=>0.8), rng=42)
mach = machine(oversampler)

# Provide the data to transform (there is nothing to fit)
Xover, yover = transform(mach, X, y)
```
The `MLJ` interface is only supported for table inputs. Read more about the interface [here]().

# TableTransforms Interface

This interface assumes that the input is one table `Xy` and that `y` is one of the columns. Hence, an integer `y_ind`
    must be specified to the constructor to specify which column `y` is followed by other keyword arguments. 
    Only `Xy` is provided while applying the transform.

```julia
using Imbalance
using TableTransforms

# Generate imbalanced data
num_rows = 200
num_features = 5
y_ind = 3
Xy, _ = generate_imbalanced_data(num_rows, num_features; 
                                 probs=[0.5, 0.2, 0.3], insert_y=y_ind, rng=42)

# Initiate Random Oversampler model
oversampler = ROSE_t(y_ind; s=0.3, ratios=Dict(0=>1.0, 1=> 0.9, 2=>0.8), rng=42)
Xyover = Xy |> oversampler                              
Xyover, cache = TableTransforms.apply(oversampler, Xy)    # equivalently
```
The `reapply(oversampler, Xy, cache)` method from `TableTransforms` simply falls back to `apply(oversample, Xy)` and the `revert(oversampler, Xy, cache)`
reverts the transform by removing the oversampled observations from the table.


# References

[1] G Menardi, N. Torelli, “Training and assessing classification rules with imbalanced data,” 
Data Mining and Knowledge Discovery, 28(1), pp.92-122, 2014.

"""
function rose(
    X::AbstractMatrix{<:AbstractFloat},
    y::AbstractVector;
    s::AbstractFloat = 1.0,
    ratios = 1.0,
    rng::Union{AbstractRNG,Integer} = default_rng(),
)
    if s < 0.0
        throw(ERR_NEG_S(s))
    end
    rng = rng_handler(rng)
    Xover, yover = generic_oversample(X, y, rose_per_class; s, ratios, rng,)
    return Xover, yover
end

# dispatch for table inputs
function rose(
    X,
    y::AbstractVector;
    s::AbstractFloat = 1.0,
    ratios = 1.0,
    rng::Union{AbstractRNG,Integer} = default_rng(),
    try_perserve_type::Bool = true
)
    Xover, yover = tablify(rose, X, y; try_perserve_type=try_perserve_type, s, ratios, rng, 
                           )
    return Xover, yover
end

# dispatch for table inputs where y is one of the columns
function rose(
    Xy,
    y_ind::Integer;
    s::AbstractFloat = 1.0,
    ratios = 1.0,
    rng::Union{AbstractRNG,Integer} = default_rng(),
    try_perserve_type::Bool = true,
)
    return tablify(rose, Xy, y_ind;try_perserve_type=try_perserve_type,  s, ratios, rng)
end
