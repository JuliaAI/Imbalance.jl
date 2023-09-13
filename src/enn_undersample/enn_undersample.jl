"""
Compute a boolean filter according to a keep_condition to implement edited nearest neighbors.

# Arguments
- X: A matrix where each row is treated as an observation
- y: A vector of labels corresponding to the observations
- k: The number of neighbors to consider while enforcing `keep_condition`
- keep_condition="mode:: The condition that leads to cleaning a point upon violation. Takes one of "exists", "mode", "only mode" and "all"
    - "exists": the point has at least one neighbor from the same class
    - "mode": the class of the point is one of the most frequent classes of the neighbors (there may be many)
    - "only mode": the class of the point is the single most frequent class of the neighbors
    - "all": the class of the point is the same as all the neighbors

# Returns
- A boolean filter that can be used to filter the data to remove the points violating "keep_condition"
"""
function compute_enn_filter(
    X::AbstractMatrix{<:Real},
    y::AbstractVector,
    k::Integer,
    keep_condition::AbstractString,
)
    tree = BallTree(X)
    # Find KNN over the whole data
    knn_map, _ = knn(tree, X, k + 1, true)
    # Convert to matrix
    knn_matrix = hcat(knn_map...)[2:end, :]
    # find the labels of each neighbor
    knn_matrix_labels = y[knn_matrix]
    # find the points to be filtered out (a binary vector to index with)
    modes_per_point = [modes(col) for col in eachcol(knn_matrix_labels)]
    bool_filter = ones(Bool, length(y))

    for i in 1:length(y)
        if keep_condition == "only mode"
            bool_filter[i] = (length(modes_per_point[i]) == 1 && y[i] in modes_per_point[i])
        elseif keep_condition == "mode"
            bool_filter[i] = (y[i] in modes_per_point[i])
        elseif keep_condition == "exists"
            bool_filter[i] = (y[i] in knn_matrix_labels[:, i])
        else
            neigh_labels = unique(knn_matrix_labels[:, i])
            bool_filter[i] = (y[i] in neigh_labels && length(neigh_labels) == 1)
        end
    end

    return BitVector(bool_filter)
end

"""
    enn_undersample(
        X, y; k = 5, keep_condition = "mode",
	    min_ratios = 1.0, force_min_ratios = false,
        rng = default_rng(), try_preserve_type=true
    )

# Description

Undersample a dataset by cleaning points that violate a certain condition such as
    belonging to a different class compared to the majority of the neighbors as proposed in [1].

# Positional Arguments

$(COMMON_DOCS["INPUTS"])

# Keyword Arguments

$(COMMON_DOCS["K-FULL"])

- `keep_condition::AbstractString="mode"`: The condition that leads to cleaning a point upon violation. Takes one of "exists", "mode", "only mode" and "all"
    - `"exists"`: the point has at least one neighbor from the same class
    - `"mode"`: the class of the point is one of the most frequent classes of the neighbors (there may be many)
    - `"only mode"`: the class of the point is the single most frequent class of the neighbors
    - `"all"`: the class of the point is the same as all the neighbors


$(COMMON_DOCS["MIN-RATIOS-UNDERSAMPLE"])

$(COMMON_DOCS["FORCE-MIN-RATIOS"])

$(COMMON_DOCS["RNG"])

$(COMMON_DOCS["TRY_PERSERVE_TYPE"])

# Returns

$(COMMON_DOCS["OUTPUTS-UNDER"])


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

# apply enn undersampling
X_under, y_under = enn_undersample(X, y; k=3, keep_condition="only mode", 
                                   min_ratios=0.5, rng=42)
julia> checkbalance(y_under)
2: ▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇ 10 (37.0%) 
1: ▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇ 10 (37.0%) 
0: ▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇ 27 (100.0%) 
```

# MLJ Model Interface

Simply pass the keyword arguments while initiating the `ENNUndersampler` model and pass the 
    positional arguments to the `transform` method. 

```julia
using MLJ
ENNUndersampler = @load ENNUndersampler pkg=Imbalance

# Wrap the model in a machine
undersampler = ENNUndersampler(k=5, min_ratios=1.0, rng=42)
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

# Initiate ENN Undersampler model
undersampler = ENNUndersampler(y_ind; min_ratios=Dict(0=>1.0, 1=> 0.9, 2=>0.8), rng=42)
Xy_under = Xy |> undersampler                    
Xy_under, cache = TableTransforms.apply(undersampler, Xy)    # equivalently
```
The `reapply(undersampler, Xy, cache)` method from `TableTransforms` simply falls back to `apply(undersample, Xy)` and the `revert(undersampler, Xy, cache)`
is not supported.

# References
[1] Dennis L Wilson. Asymptotic properties of nearest neighbor rules using edited data. 
	IEEE Transactions on Systems, Man, and Cybernetics, pages 408–421, 1972.
"""
function enn_undersample(
    X::AbstractMatrix{<:Real},
    y::AbstractVector;
    k::Integer = 5,
    keep_condition::AbstractString = "mode",
    min_ratios = 1.0,
    force_min_ratios = false,
    rng::Union{AbstractRNG, Integer} = default_rng(),
)
    rng = rng_handler(rng)
    check_k(k, size(X, 1))
    (keep_condition in ["exists", "mode", "only mode", "all"]) ||
        throw("keep_condition must be one of: exists, mode, only mode, all")
    X = transpose(X)
    filter = compute_enn_filter(X, y, k, keep_condition)
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
function enn_undersample(
    X,
    y::AbstractVector;
    k::Integer = 5,
    keep_condition::AbstractString = "mode",
    min_ratios = 1.0,
    force_min_ratios = false,
    rng::Union{AbstractRNG, Integer} = default_rng(),
    try_perserve_type::Bool = true,
)
    X_under, y_under = tablify(
        enn_undersample,
        X,
        y;
        try_perserve_type = try_perserve_type,
        encode_func = generic_encoder,
        decode_func = generic_decoder,
        k,
        keep_condition,
        min_ratios,
        force_min_ratios,
        rng,
    )
    return X_under, y_under
end

# dispatch for table inputs where y is one of the columns
function enn_undersample(
    Xy,
    y_ind::Integer;
    k::Integer = 5,
    keep_condition::AbstractString = "mode",
    min_ratios = 1.0,
    force_min_ratios = false,
    rng::Union{AbstractRNG, Integer} = default_rng(),
    try_perserve_type::Bool = true,
)
    Xy_under = tablify(
        enn_undersample,
        Xy,
        y_ind;
        try_perserve_type = try_perserve_type,
        encode_func = generic_encoder,
        decode_func = generic_decoder,
        k,
        keep_condition,
        min_ratios,
        force_min_ratios,
        rng,
    )
    return Xy_under
end
