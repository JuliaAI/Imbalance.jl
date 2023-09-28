
"""
This function is only called when N>1 and checks whether 0<m<N or not. If m<0, it throws an error.
and if m>=N, it warns the user and sets m=N-1.

# Arguments
- `k`: Number of nearest neighbors to consider
- `n`: Number of observations in the data

# Returns
-  Number of nearest neighbors to consider
"""
function check_m(m, N)
    if m < 1
        throw(ERR_NONPOS_M(m))
    end
    if m >= N
        @warn WRN_M_TOO_BIG(m, N)
        m = N - 1
    end
    return m
end

"""
Compute a boolean filter according to filter (X, y) points that satisfy the BorderlineSMOTE1 condition.

# Arguments
- X: A matrix where each row is treated as an observation
- y: A vector of labels corresponding to the observations
- m: The number of neighbors to consider while checking the BorderlineSMOTE1 condition. In this, a point may participate
    in oversampling iff the number of neighbors that belong to its class is in \$(0, m/2]\$

# Returns
- A boolean filter that can be used to filter the data to remove the points violating the BorderlineSMOTE1 condition.
"""
function borderline1_filter(X, y; m=5)
    m = check_m(m, length(y))
    tree = BallTree(X)
    knn_map, _ = knn(tree, X, m + 1, true)
    knn_matrix = hcat(knn_map...)[2:end, :]
    # m * num_points mapping from point index to neighbor labels
    knn_matrix_labels = y[knn_matrix]
    bool_filter = ones(Bool, length(y))
    for i in eachindex(y)
        num_friendly_neighbors = sum(knn_matrix_labels[:, i] .== y[i])
        # borderline_smote1 condition
        bool_filter[i] = 0 < num_friendly_neighbors  <= m/2
    end
    return BitVector(bool_filter)
end


"""
    borderline_smote1(
        X, y;
        m=5, k=5, ratios=nothing, rng=default_rng(),
        try_perserve_type=true
    )

# Description
Oversamples a dataset using borderline SMOTE1 algorithm to 
    correct for class imbalance as presented in [1]

# Positional Arguments

$(COMMON_DOCS["INPUTS"])

# Keyword Arguments

- `m::Integer=5`: The number of neighbors to consider while checking the BorderlineSMOTE1 condition. In this, a point may participate
    in oversampling if and only if the number of neighbors that belong to its class is in \$(0, m/2]\$. Should be within the range 
   `0 < m < N` where N is the number of observations in the data. It will be automatically set to `N-1` if `N ≤ m`.

- `k::Integer=5`: Number of nearest neighbors to consider in the SMOTE part of the algorithm. Should be within the range
   `0 < k < n` where n is the number of observations in the smallest class. It will be automatically set to
   `n-1` for any class where `n ≤ k`.

$(COMMON_DOCS["RATIOS"])

$(COMMON_DOCS["RNG"])

$(COMMON_DOCS["TRY_PERSERVE_TYPE"])

# Returns

$(COMMON_DOCS["OUTPUTS"])


# Example

```@repl
using Imbalance

# set probability of each class
probs = [0.5, 0.2, 0.3]                         
num_rows, num_continuous_feats = 100, 5
# generate a table and categorical vector accordingly
X, y = generate_imbalanced_data(num_rows, num_continuous_feats; 
                                probs, rng=42)            

julia> Imbalance.checkbalance(y)
1: ▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇ 19 (39.6%) 
2: ▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇ 33 (68.8%) 
0: ▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇ 48 (100.0%) 

# apply BorderlineSMOTE1
Xover, yover = borderline_smote1(X, y; m = 3, 
               k = 5, ratios = Dict(0=>1.0, 1=> 0.9, 2=>0.8), rng = 42)

julia> Imbalance.checkbalance(y)
2: ▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇ 38 (79.2%) 
1: ▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇ 43 (89.6%) 
0: ▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇ 48 (100.0%) 
```

# MLJ Model Interface

Simply pass the keyword arguments while initiating the `BorderlineSMOTE1` model and pass the 
    positional arguments to the `transform` method. 

```julia
using MLJ
BorderlineSMOTE1 = @load BorderlineSMOTE1 pkg=Imbalance

# Wrap the model in a machine
oversampler = BorderlineSMOTE1(m=3, k=5, ratios=Dict(0=>1.0, 1=> 0.9, 2=>0.8), rng=42)
mach = machine(oversampler)

# Provide the data to transform (there is nothing to fit)
Xover, yover = transform(mach, X, y)
```
You can read more about this `MLJ` interface [here]().



# TableTransforms Interface

This interface assumes that the input is one table `Xy` and that `y` is one of the columns. Hence, an integer `y_ind`
    must be specified to the constructor to specify which column `y` is followed by other keyword arguments. 
    Only `Xy` is provided while applying the transform.

```julia
using Imbalance
using Imbalance.TableTransforms

# Generate imbalanced data
num_rows = 200
num_features = 5
y_ind = 3
Xy, _ = generate_imbalanced_data(num_rows, num_features; 
                                 probs=[0.5, 0.2, 0.3], insert_y=y_ind, rng=42)

# Initiate BorderlineSMOTE1 Oversampler model
oversampler = BorderlineSMOTE1(y_ind; m=3, k=5, 
              ratios=Dict(0=>1.0, 1=> 0.9, 2=>0.8), rng=42)
Xyover = Xy |> oversampler                              
Xyover, cache = TableTransforms.apply(oversampler, Xy)    # equivalently
```
The `reapply(oversampler, Xy, cache)` method from `TableTransforms` simply falls back to `apply(oversample, Xy)` and the `revert(oversampler, Xy, cache)`
reverts the transform by removing the oversampled observations from the table.


# References
[1] Han, H., Wang, W.-Y., & Mao, B.-H. (2005). Borderline-SMOTE: A new over-sampling method in imbalanced data sets learning. 
    In D.S. Huang, X.-P. Zhang, & G.-B. Huang (Eds.), Advances in Intelligent Computing (pp. 878-887). Springer. 
"""
function borderline_smote1(
	X::AbstractMatrix{<:AbstractFloat},
	y::AbstractVector;
    m::Integer = 5,
	k::Integer = 5,
	ratios = 1.0,
	rng::Union{AbstractRNG, Integer} = default_rng(),
    try_perserve_type::Bool = true,
)
    # this function adjust generic_oversampling to use in borderline smote
    rng = rng_handler(rng)
    X = transpose(X)
    filter = borderline1_filter(X, y; m)
    X1, y1 = X[:, filter], y[filter]       # filtered points should be oversampled

    # give the user an idea about the filtered data.
    y1_stats = match(r"\((.*?)\)", string(countmap(y1))).captures[1]
    @info "After filtering, the mapping from each class to number of borderline points is ($y1_stats)."
    length(y1) == 0 && throw(ERR_NO_BORDERLINE)
    
    # Get maps from labels to indices and the needed counts
    label_inds = group_inds(y1)
    extra_counts = get_class_counts(y, ratios)

    # Apply oversample per class on each set of points belonging to the same class
    p = Progress(length(label_inds))
    for (label, inds) in label_inds
        !(label in keys(extra_counts)) && continue
        # Get points belonging to class
        X_label1 = @view X1[:, inds]   
        # How many points does it need?
        n = extra_counts[label]
        n == 0 && continue
        # Generate the n needed new points
        Xnew = smote_per_class(X_label1, n; k, rng)
        # Generate the corresponding labels
        ynew = fill(label, size(Xnew, 2))
        X = hcat(X, Xnew)
        y = vcat(y, ynew)
        next!(p; showvalues = [(:class, label)])
    end
    yover = y
    Xover = transpose(X)
    return Xover, yover
end

# dispatch for table inputs
function borderline_smote1(
	X,
	y::AbstractVector;
    m::Integer = 5,
	k::Integer = 5,
	ratios = 1.0,
	rng::Union{AbstractRNG, Integer} = default_rng(),
    try_perserve_type::Bool = true,
)
    Xover, yover = tablify(borderline_smote1, X, y; try_perserve_type=try_perserve_type,  m, k, ratios, rng)
    return Xover, yover
end

# dispatch for table inputs where y is one of the columns
function borderline_smote1(
    Xy,
    y_ind::Integer;
    m::Integer = 5,
	k::Integer = 5,
	ratios = 1.0,
	rng::Union{AbstractRNG, Integer} = default_rng(),
    try_perserve_type::Bool = true,
)
    Xyover = tablify(borderline_smote1, Xy, y_ind; try_perserve_type=try_perserve_type, m, k, ratios, rng)
    return Xyover
end