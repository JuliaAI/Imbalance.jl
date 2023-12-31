"""
Assuming that all the observations in the observation matrix X belong to the same class, undersample
the observations using clustering undersampling

# Arguments
- `X`: A matrix where each column is an observation of floats
- `n`: Number of observations in the undersampled dataset
- `mode::AbstractString="nearest`: If `"center"` then the undersampled data will consist of the centriods of 
    each cluster found; if `"nearest"` then it will consist of the nearest neighbor of each centroid.
- `maxiter::Integer=100`: Maximum number of iterations to run K-means
- `rng::Integer=42`: Random number generator seed
# Returns
- `Xnew`: A matrix containing the undersampled observations
"""
function cluster_undersample_per_class(
    X::AbstractMatrix{<:Real},
    n::Integer;
    mode::AbstractString = "nearest",
    maxiter::Integer = 100,
    rng::Union{AbstractRNG,Integer} = default_rng()
)
    (mode in ["center", "nearest"]) ||
        throw((ERR_INVALID_MODE))
    # to undersample down to n points find k=n clusters
    result = kmeans(X, n; maxiter, rng)
    # the n cluster centers are the undersampled points
    X_new = result.centers
    # unless mode is "nearest" where we seek the nearest neighbor of each center
    if mode == "nearest"
        tree = BallTree(X)
        keep_inds, _ = knn(tree, X_new, 1, true)
        keep_inds = vcat(keep_inds...)      # flatten
        X_new = X[:, keep_inds]
    end
    return X_new
end

"""
    cluster_undersample(
        X, y; 
        mode= "nearest", ratios = 1.0, maxiter = 100,
        rng=default_rng(), try_preserve_type=true
    )


# Description

Undersample a dataset using clustering undersampling as presented in [1] using K-means as the clustering algorithm.

# Positional Arguments

$(COMMON_DOCS["INPUTS"])

# Keyword Arguments

- `mode::AbstractString="nearest`: If `"center"` then the undersampled data will consist of the centriods of 
    each cluster found; meanwhile, if `"nearest"` then it will consist of the nearest neighbor of each centroid.

$(COMMON_DOCS["RATIOS-UNDERSAMPLE"])

- `maxiter::Integer=100`: Maximum number of iterations to run K-means

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

# apply cluster_undersampling
X_under, y_under = cluster_undersample(X, y; mode="nearest", 
                                       ratios=Dict(0=>1.0, 1=> 1.0, 2=>1.0), rng=42)
                                       
julia> Imbalance.checkbalance(y_under; ref="minority")
0: ▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇ 19 (100.0%) 
2: ▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇ 19 (100.0%) 
1: ▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇ 19 (100.0%) 
```

# MLJ Model Interface

Simply pass the keyword arguments while initiating the `ClusterUndersampler` model and pass the 
    positional arguments `X, y` to the `transform` method. 

```julia
using MLJ
ClusterUndersampler = @load ClusterUndersampler pkg=Imbalance

# Wrap the model in a machine
undersampler = ClusterUndersampler(mode="nearest", 
                                   ratios=Dict(0=>1.0, 1=> 1.0, 2=>1.0), rng=42)
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

# Initiate ClusterUndersampler model
undersampler = ClusterUndersampler(y_ind; mode="nearest", 
                                   ratios=Dict(0=>1.0, 1=> 0.9, 2=>0.8), rng=42)
Xy_under = Xy |> undersampler                    
Xy_under, cache = TableTransforms.apply(undersampler, Xy)    # equivalently
```
The `reapply(undersampler, Xy, cache)` method from `TableTransforms` simply falls back to `apply(undersample, Xy)` and the `revert(undersampler, Xy, cache)`
is not supported.


# Illustration
A full basic example along with an animation can be found [here](https://githubtocolab.com/JuliaAI/Imbalance.jl/blob/dev/examples/undersample_cluster.ipynb). 
    You may find more practical examples in the [tutorial](https://juliaai.github.io/Imbalance.jl/dev/examples/) 
    section which also explains running code on Google Colab.

# References
[1] Wei-Chao, L., Chih-Fong, T., Ya-Han, H., & Jing-Shang, J. (2017). 
    Clustering-based undersampling in class-imbalanced data. Information Sciences, 409–410, 17–26.
"""
function cluster_undersample(
    X::AbstractMatrix{<:Real},
    y::AbstractVector;
    mode::AbstractString = "nearest",
    ratios = 1.0,
    maxiter::Integer = 100,
    rng::Union{AbstractRNG,Integer} = default_rng(),
)
    rng = rng_handler(rng)
    X_under, y_under =
        generic_undersample(X, y, cluster_undersample_per_class; ratios, mode, maxiter, rng)
    return X_under, y_under
end

# dispatch for when X is a table
function cluster_undersample(
    X,
    y::AbstractVector;
    mode::AbstractString = "nearest",
    ratios = 1.0,
    maxiter::Integer = 100,
    rng::Union{AbstractRNG,Integer} = default_rng(),
    try_preserve_type::Bool = true,
)
    X_under, y_under = tablify(
        cluster_undersample,
        X,
        y;
        try_preserve_type = try_preserve_type,
        encode_func = generic_encoder,
        decode_func = generic_decoder,
        mode,
        ratios,
        maxiter,
        rng,
    )
    return X_under, y_under
end

# dispatch for table inputs where y is one of the columns
function cluster_undersample(
    Xy,
    y_ind::Integer;
    mode::AbstractString = "nearest",
    ratios = 1.0,
    maxiter::Integer = 100,
    rng::Union{AbstractRNG,Integer} = default_rng(),
    try_preserve_type::Bool = true,
)
    Xy_under = tablify(
        cluster_undersample,
        Xy,
        y_ind;
        try_preserve_type = try_preserve_type,
        encode_func = generic_encoder,
        decode_func = generic_decoder,
        mode,
        ratios,
        maxiter,
        rng,
    )
    return Xy_under
end
