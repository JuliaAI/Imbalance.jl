"""
Compute a boolean filter that eliminates any point that is part of a tomek link in the data.

# Arguments
- X: A matrix where each row is treated as an observation
- y: A vector of labels corresponding to the observations

# Returns
- A boolean filter that can be used to filter the data to remove the points
"""
function compute_tomek_filter(X::AbstractMatrix{<:Real}, y::AbstractVector)
    tree = BallTree(X)
    # Find KNN over the whole data
    knn_map, _ = knn(tree, X, 2, true)
    # prepare filter
    bool_filter = ones(Bool, size(knn_map, 1))
    for (i, neigh_inds) in enumerate(knn_map)
        # get the point's nearest neighbor
        nn = neigh_inds[2]
        # if the point has a label different than its nearest neighbor
        # and it itself is its nearest neighbor's nearest neighbor
        # then it should be cleaned.
        if y[i] != y[nn] && knn_map[nn][2] == i
            bool_filter[i] = false
        end
    end
    return BitVector(bool_filter)
end

"""
    tomek_undersample(
        X, y;
	    min_ratios = 1.0, force_min_ratios = false,
        rng = default_rng(), try_preserve_type=true
    )

# Description

Undersample a dataset by removing ("cleaning") any point that is part of a tomek link in the data. 
	Tomek links are presented in [1].

# Positional Arguments

$(COMMON_DOCS["INPUTS"])

# Keyword Arguments

$(COMMON_DOCS["MIN-RATIOS-UNDERSAMPLE"])

$(COMMON_DOCS["FORCE-MIN-RATIOS"])

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
                                min_sep=0.01, stds=[3.0 3.0 3.0], class_probs, rng=42)   

julia> Imbalance.checkbalance(y; ref="minority")
1: ▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇ 19 (100.0%) 
2: ▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇ 33 (173.7%) 
0: ▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇ 48 (252.6%) 

# apply tomek undersampling
X_under, y_under = tomek_undersample(X, y; min_ratios=1.0, rng=42)

julia> Imbalance.checkbalance(y_under; ref="minority")
1: ▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇ 19 (100.0%) 
2: ▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇ 22 (115.8%) 
0: ▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇ 36 (189.5%)
```

# MLJ Model Interface

Simply pass the keyword arguments while initiating the `TomekUndersampler` model and pass the 
    positional arguments `X, y` to the `transform` method. 

```julia
using MLJ
TomekUndersampler = @load TomekUndersampler pkg=Imbalance

# Wrap the model in a machine
undersampler = TomekUndersampler(min_ratios=1.0, rng=42)
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
                                min_sep=0.01, stds=[3.0 3.0 3.0], class_probs, rng=42)   

# Initiate TomekUndersampler model
undersampler = TomekUndersampler(y_ind; min_ratios=1.0, rng=42)
Xy_under = Xy |> undersampler                    
Xy_under, cache = TableTransforms.apply(undersampler, Xy)    # equivalently
```
The `reapply(undersampler, Xy, cache)` method from `TableTransforms` simply falls back to `apply(undersample, Xy)` and the `revert(undersampler, Xy, cache)`
is not supported.

# Illustration
A full basic example along with an animation can be found [here](https://githubtocolab.com/JuliaAI/Imbalance.jl/blob/dev/examples/undersample_tomek.ipynb). 
    You may find more practical examples in the [tutorial](https://juliaai.github.io/Imbalance.jl/dev/examples/) 
    section which also explains running code on Google Colab.

# References
[1] Ivan Tomek. Two modifications of cnn. IEEE Trans. Systems, Man and Cybernetics, 6:769–772, 1976.
"""
function tomek_undersample(
    X::AbstractMatrix{<:Real},
    y::AbstractVector;
    min_ratios = 1.0,
    force_min_ratios = false,
    rng::Union{AbstractRNG, Integer} = default_rng(),
    try_preserve_type::Bool = true,
)
    rng = rng_handler(rng)
    X = transpose(X)
    filter = compute_tomek_filter(X, y)
    pass_inds, is_transposed = true, true
    X_under, y_under = generic_undersample(
        X,
        y,
        generic_clean_per_class,
        filter;
        ratios = min_ratios,
        is_transposed,
        pass_inds,
        force_min_ratios,
        rng,
    )
    return X_under, y_under
end

# dispatch for when X is a table
function tomek_undersample(
    X,
    y::AbstractVector;
    min_ratios = 1.0,
    force_min_ratios = false,
    rng::Union{AbstractRNG, Integer} = default_rng(),
    try_preserve_type::Bool = true,
)
    X_under, y_under = tablify(
        tomek_undersample,
        X,
        y;
        try_preserve_type = try_preserve_type,
        encode_func = generic_encoder,
        decode_func = generic_decoder,
        min_ratios,
        force_min_ratios,
        rng,
    )
    return X_under, y_under
end

# dispatch for table inputs where y is one of the columns
function tomek_undersample(
    Xy,
    y_ind::Integer;
    min_ratios = 1.0,
    force_min_ratios = false,
    rng::Union{AbstractRNG, Integer} = default_rng(),
    try_preserve_type::Bool = true,
)
    Xy_under = tablify(
        tomek_undersample,
        Xy,
        y_ind;
        try_preserve_type = try_preserve_type,
        encode_func = generic_encoder,
        decode_func = generic_decoder,
        min_ratios,
        force_min_ratios,
        rng,
    )
    return Xy_under
end
