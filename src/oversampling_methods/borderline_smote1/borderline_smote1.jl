"""
Choose a random point from the given observations matrix `X` and generate a new point that 
randomly lies in the line joining the random point and randomly one of its k-nearest neighbors. 

# Arguments
- `X`: A matrix where each column is an observation
- `knn_map`: A vector of vectors mapping each element in X by index to its nearest neighbors' indices
- `class_filter`: A bit vector that decides which points from the class data may generate new points using SMOTE
- `rng`: Random number generator

# Returns
- `x_new`: A new observation generated by SMOTE
"""
function generate_new_borderline_smote1_point(
    X::AbstractMatrix{<:AbstractFloat},
    knn_map,
    class_filter;
    rng::AbstractRNG,
)
    # 1. Choose a random point from X (by index)
    ind = rand(rng, findall(class_filter))
    x_rand = X[:, ind]
    # 2. Choose a random point from its k-nearest neighbors 
    x_rand_neigh = get_random_neighbor(X, ind, knn_map; rng)
    # 3. Generate a new point that randomly lies in the line between them
    x_new = get_collinear_point(x_rand, x_rand_neigh; rng)
    return x_new
end

"""
Assuming that all the observations in the observation matrix X belong to the same class,
use SMOTE to generate `n` new observations for that class.

# Arguments
- `X`: A matrix where each row is an observation
- `n`: Number of new observations to generate
- `inds`: The indices of points belonging to the current class
- `filter`: A bit vector that decides which points from the whole data may generate new points using SMOTE
- `k`: Number of nearest neighbors to consider. Must be less than the 
    number of observations in `X`
- `rng`: Random number generator

# Returns
- `Xnew`: A matrix where each row is a new observation generated by SMOTE
"""
function borderline_smote1_per_class(
    X::AbstractMatrix{<:AbstractFloat},
    n::Integer,
    inds::AbstractVector{<:Integer},
    filter::AbstractVector{Bool};
    k::Integer = 5,
    rng::AbstractRNG = default_rng(),
)
    # Can't draw lines if there are no neighbors
    n_class = size(X, 2)
    n_class == 1 && (@warn WRN_SINGLE_OBS; return Array{Float64}(undef, size(X, 1), 0))

    # Automatically fix k if needed
    k = check_k(k, n_class)

    # Build KDTree for KNN
    tree = KDTree(X)
    knn_map, _ = knn(tree, X, k + 1, true)
    
    # get class filter (borderline points)
    class_filter = filter[inds]         
    sum(class_filter) == 0 && (@warn WRN_NO_BORDERLINE_CLASS; return Array{Float64}(undef, size(X, 1), 0))

    # Generate n new observations
    Xnew = zeros(Float32, size(X, 1), n)
    p = Progress(n)
    for i=1:n
        Xnew[:, i] = generate_new_borderline_smote1_point(X, knn_map, class_filter; rng)
        next!(p)
    end
    return Xnew
end


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
- m: The number of neighbors to consider while checking the BorderlineSMOTE1 condition. 
- `verbosity::Integer=1`: Whenever higher than `0` info regarding the points that will participate in oversampling is logged.

# Returns
- A boolean filter that can be used to filter the data to remove the points violating the BorderlineSMOTE1 condition.
"""
function borderline1_filter(X, y; m=5, verbosity=1)
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
    y1 = y[bool_filter]
    y1_stats = match(r"\((.*?)\)", string(countmap(y1))).captures[1]
    verbosity > 0 && @info INFO_BORDERLINE_PTS(y1_stats)
    length(y1) == 0 && throw(ERR_NO_BORDERLINE)
    return BitVector(bool_filter)
end


"""
    borderline_smote1(
        X, y;
        m=5, k=5, ratios=nothing, rng=default_rng(),
        try_perserve_type=true, verbosity=1
    )

# Description
Oversamples a dataset using borderline SMOTE1 algorithm to 
    correct for class imbalance as presented in [1]

# Positional Arguments

$(COMMON_DOCS["INPUTS"])

# Keyword Arguments

- `m::Integer=5`: The number of neighbors to consider while checking the BorderlineSMOTE1 condition. Should be within the range 
   `0 < m < N` where N is the number of observations in the data. It will be automatically set to `N-1` if `N ≤ m`.

- `k::Integer=5`: Number of nearest neighbors to consider in the SMOTE part of the algorithm. Should be within the range
   `0 < k < n` where n is the number of observations in the smallest class. It will be automatically set to
   `n-1` for any class where `n ≤ k`.

$(COMMON_DOCS["RATIOS"])

$(COMMON_DOCS["RNG"])

$(COMMON_DOCS["TRY_PERSERVE_TYPE"])

- `verbosity::Integer=1`: Whenever higher than `0` info regarding the points that will participate in oversampling is logged.

# Returns

$(COMMON_DOCS["OUTPUTS"])


# Example

```@repl
using Imbalance

# set probability of each class
class_probs = [0.5, 0.2, 0.3]                         
num_rows, num_continuous_feats = 100, 5
# generate a table and categorical vector accordingly
X, y = generate_imbalanced_data(num_rows, num_continuous_feats; 
                                class_probs, rng=42)            

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
                                 class_probs=[0.5, 0.2, 0.3], insert_y=y_ind, rng=42)

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
    verbosity::Integer = 1
)
    # this function is a variation on generic_oversampling to use in borderline smote
    rng = rng_handler(rng)
    X = transpose(X)
    filter = borderline1_filter(X, y; m, verbosity)
    Xover, yover = generic_oversample(X, y, borderline_smote1_per_class, filter; pass_inds=true, is_transposed=true, k, ratios, rng)
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
    verbosity::Integer = 1
)
    Xover, yover = tablify(borderline_smote1, X, y; try_perserve_type=try_perserve_type,  m, k, ratios, rng, verbosity)
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
    verbosity::Integer = 1
)
    Xyover = tablify(borderline_smote1, Xy, y_ind; try_perserve_type=try_perserve_type, m, k, ratios, rng, verbosity)
    return Xyover
end