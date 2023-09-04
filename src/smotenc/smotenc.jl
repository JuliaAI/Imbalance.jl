
# SMOTE-NC uses KNN with a modified distance metric. Refer to 
# "SMOTE: Synthetic Minority Over-sampling Technique" by Chawla et al. (2002), pg. 351. 
include("../distance_metrics/penalized_euclidean.jl")

"""
Label encode and decode each column in a given table X
"""
smotenc_encoder(X) = generic_encoder(X; error_checker=check_scitypes_smotenc,  return_cat_inds = true)
smotenc_decoder(X, d) = generic_decoder(X, d)

"""
Check that all columns are categorical . If not, throw an error.

# Arguments
- `ncols`: Number of columns
- `cat_inds`: Indices of categorical columns
- `cont_inds`: Indices of continuous columns
- `types`: Types of each column

"""
function check_scitypes_smotenc(ncols, cat_inds, cont_inds, types)
    bad_cols = setdiff(1:ncols, vcat(cat_inds, cont_inds))
    if !isempty(bad_cols)
        throw(ArgumentError(ERR_BAD_MIXED_COL_TYPES(bad_cols, types[bad_cols])))
    end
end


"""
Given a matrix of observations `X`, find the median of the standard deviations of the
continuous features of the observations and return that as the penalty.

# Arguments
- `X`: A matrix where each row is an observation
- `cont_inds`: A vector of indices of the continuous features

# Returns
- `σₘ^2`: The penalty term that modifies the distance metric

"""
function get_penalty(X::AbstractMatrix{<:AbstractFloat}, cont_inds::AbstractVector{<:Int})
    # simply compute the penalty as described
    Xcont = @view X[cont_inds, :]
    σs = vec(std(Xcont, dims = 2))
    σₘ = median(σs)
    return σₘ^2
end



"""
Choose a random point from the given observations matrix `X` and generate a new point that
in terms of the continuous part, randomly lies in the line joining the random point and 
randomly one of its k-nearest neighbors and in terms of the categorical part, the new point
has the mode of the categorical part of the k-nearest neighbors of the random point.

# Arguments
- `X`: A matrix where each row is an observation
- `tree`: A k-d tree representation of the observations matrix X
- `cont_inds`: A vector of indices of the continuous features
- `cat_inds`: A vector of indices of the categorical features
- `k`: Number of nearest neighbors to consider
- `rng`: Random number generator

# Returns
- `x_new`: A new observation generated by SMOTE
"""
function generate_new_smotenc_point(
    X::AbstractMatrix{<:AbstractFloat},
    cont_inds::AbstractVector{<:Int},
    cat_inds::AbstractVector{<:Int},
    knn_map;
    rng::AbstractRNG,
)
    # 1. Choose a random point from X (by index)
    ind = rand(rng, 1:size(X, 2))
    x_rand = X[:, ind]
    x_randneigh = get_random_neighbor(X, ind, knn_map; rng)

    # find the continuous part of new point (by choosing a random point along the line)
    x_rand_cont = @view x_rand[cont_inds]
    x_randneigh_cont = @view x_randneigh[cont_inds]
    x_new_cont = get_collinear_point(x_rand_cont, x_randneigh_cont; rng = rng)

    # find the categorical part of new point (by voting among all neighbors)
    Xneighs = X[:, knn_map[ind][2:end]]
    Xneighs_cat = @view Xneighs[cat_inds, :]
    x_new_cat = get_neighbors_mode(Xneighs_cat, rng)

    # make the final vector
    x_new = fill(0.0, size(X, 1))
    x_new[cont_inds] = x_new_cont
    x_new[cat_inds] = x_new_cat

    return x_new
end



"""
Assuming that all the observations in the observation matrix X belong to the same class,
use SMOTE-NC to generate `n` new observations for that class.

# Arguments
- `X`: A matrix where each row is an observation
- `n`: Number of new observations to generate
- `k`: Number of nearest neighbors to consider.
- `cont_inds`: A vector of indices of the continuous features
- `cat_inds`: A vector of indices of the categorical features
- `knn_tree`: Decides the tree used in KNN computations. Either "Brute" or "Ball".
    BallTree is much faster but may lead to innacurate results.
- `rng`: Random number generator

# Returns
- `Xnew`: A matrix where each row is a new observation generated by SMOTE
"""
function smotenc_per_class(
    X::AbstractMatrix{<:AbstractFloat},
    n::Integer,
    cont_inds::AbstractVector{<:Int},
    cat_inds::AbstractVector{<:Int};
    k::Integer = 5,
    knn_tree::AbstractString = "Brute",
    rng::AbstractRNG = default_rng(),
)
    # Can't draw lines if there are no neighbors
    n_class = size(X, 2)
    n_class == 1 && (@warn WRN_SINGLE_OBS; return X)

    # Automatically fix k if needed
    k = check_k(k, n_class)

    # Build a KNN tree with the modified distance metric
    p = get_penalty(X, cont_inds)
    metric = EuclideanWithPenalty(p, cont_inds, cat_inds)
    (knn_tree ∈ ["Ball", "Brute"]) || throw(ERR_WRNG_TREE(knn_tree))
    tree = (knn_tree == "Brute") ? BruteTree(X, metric) : BallTree(X, metric)
    knn_map, _ = knn(tree, X, k + 1, true)

    # Generate n new observations
    Xnew = zeros(Float32, size(X, 1), n)
    p = Progress(n)
    for i=1:n
        Xnew[:, i] = generate_new_smotenc_point(X, cont_inds, cat_inds, knn_map; rng)
        next!(p)
    end
    return Xnew
end


"""
    smotenc(
        X, y, split_ind;
        k=5, ratios=nothing, rng=default_rng(),
        try_perserve_type=true
    )

# Description

Oversamples a dataset using `SMOTE-NC` (Synthetic Minority Oversampling Techniques-Nominal Continuous) 
    algorithm to correct for class imbalance as presented in [1]. This is a variant of `SMOTE` 
    to deal with datasets with both nominal and continuous features. 

!!! warning "SMOTE-NC Assumes Continuous Features Exist"
    SMOTE-NC will not work if the dataset is purely nominal. In that case, refer to [SMOTE-N](@ref) instead.
        Meanwhile, if the dataset is purely continuous then it's equivalent to the standard [SMOTE`](@ref).

# Positional Arguments

- `X`: A matrix of floats or a table with element [scitypes](https://juliaai.github.io/ScientificTypes.jl/) that subtype `Union{Finite, Infinite}`. 
     Elements in nominal columns should subtype `Finite` (i.e., have [scitype](https://juliaai.github.io/ScientificTypes.jl/) `OrderedFactor` or `Multiclass`) and
     elements in continuous columns should subtype `Infinite` (i.e., have [scitype](https://juliaai.github.io/ScientificTypes.jl/) `Count` or `Continuous`).

- `y`: An abstract vector of labels (e.g., strings) that correspond to the observations in `X`

- `cat_inds::AbstractVector{<:Int}`: A vector of the indices of the nominal features. Supplied only if `X` is a matrix.
        Otherwise, they are inferred from the table's [scitypes](https://juliaai.github.io/ScientificTypes.jl/).


# Keyword Arguments

$(COMMON_DOCS["K"])

$(COMMON_DOCS["RATIOS"])

- `knn_tree`: Decides the tree used in KNN computations. Either "Brute" or "Ball".
    BallTree can be much faster but may lead to innacurate results.
$(COMMON_DOCS["RNG"])

$(COMMON_DOCS["TRY_PERSERVE_TYPE"])

# Returns

$(COMMON_DOCS["OUTPUTS"])

# Example
```@repl
using Imbalance
using StatsBase

# set probability of each class
probs = [0.5, 0.2, 0.3]                         
num_rows = 100
num_continuous_feats = 3
# want two categorical features with three and two possible values respectively
cat_feats_num_vals = [3, 2]

# generate a table and categorical vector accordingly
X, y = generate_imbalanced_data(num_rows, num_continuous_feats; 
                                probs, cat_feats_num_vals, rng=42)                      
julia> StatsBase.countmap(y)
Dict{CategoricalArrays.CategoricalValue{Int64, UInt32}, Int64} with 3 entries:
0 => 48
2 => 33
1 => 19

julia> ScientificTypes.schema(X).scitypes
(Continuous, Continuous, Continuous, Continuous, Continuous)
# coerce nominal columns to a finite scitype (multiclass or ordered factor)
X = coerce(X, :Column4=>Multiclass, :Column5=>Multiclass)

# apply SMOTE-NC
Xover, yover = smotenc(X, y; k = 5, ratios = Dict(0=>1.0, 1=> 0.9, 2=>0.8), rng = 42)
julia>StatsBase.countmap(yover)
Dict{CategoricalArrays.CategoricalValue{Int64, UInt32}, Int64} with 3 entries:
0 => 48
2 => 33
1 => 19
```
# MLJ Model Interface

Simply pass the keyword arguments while initiating the `SMOTENC` model and pass the 
    positional arguments (excluding `cat_inds`) to the `transform` method. 

```julia
using MLJ
SMOTEN = @load SMOTEN pkg=Imbalance

# Wrap the model in a machine
oversampler = SMOTENC(k=5, ratios=Dict(0=>1.0, 1=> 0.9, 2=>0.8), rng=42)
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
using ScientificTypes
using Imbalance.TableTransforms

# Generate imbalanced data
num_rows = 100
num_continuous_feats = 3
y_ind = 2
# generate a table and categorical vector accordingly
Xy, _ = generate_imbalanced_data(num_rows, num_continuous_feats; insert_y=y_ind,
                                probs= [0.5, 0.2, 0.3], cat_feats_num_vals=[3, 2],
                                 rng=42)  

# Table must have only finite or continuous scitypes                                
Xy = coerce(Xy, :Column2=>Multiclass, :Column5=>Multiclass, :Column6=>Multiclass)

# Initiate Random Oversampler model
oversampler = SMOTENC(y_ind; k=5, ratios=Dict(1=>1.0, 2=> 0.9, 3=>0.9), rng=42)
Xyover = Xy |> oversampler                               
Xyover, cache = TableTransforms.apply(oversampler, Xy)    # equivalently
```

# References
[1] N. V. Chawla, K. W. Bowyer, L. O.Hall, W. P. Kegelmeyer,
“SMOTE: synthetic minority over-sampling technique,”
Journal of artificial intelligence research, 321-357, 2002.

"""
function smotenc(
    X::AbstractMatrix{<:AbstractFloat},
    y::AbstractVector,
    cat_inds::AbstractVector{<:Int};
    k::Integer = 5,
    ratios = 1.0,
    knn_tree::AbstractString = "Brute",
    rng::Union{AbstractRNG,Integer} = default_rng(),
)
    rng = rng_handler(rng)
    # implictly infer the continuous indices
    cont_inds = setdiff(1:size(X, 2), cat_inds)
    Xover, yover =
        generic_oversample(X, y, smotenc_per_class, cont_inds, cat_inds; ratios, k, knn_tree, rng)
    return Xover, yover
end

# dispatch for when X is a table
function smotenc(
    X,
    y::AbstractVector;
    k::Integer = 5,
    ratios = 1.0,
    knn_tree::AbstractString = "Brute",
    rng::Union{AbstractRNG,Integer} = default_rng(),
    try_perserve_type::Bool = true,
)
    Xover, yover = tablify(
        smotenc,
        X,
        y;
        try_perserve_type=try_perserve_type,
        encode_func = smotenc_encoder,
        decode_func = smotenc_decoder,
        k,
        ratios,
        knn_tree,
        rng,
    )
    return Xover, yover
end

# dispatch for when X is a table and y is one of its columns
function smotenc(
    Xy,
    y_ind::Integer;
    k::Integer = 5,
    ratios = 1.0,
    knn_tree::AbstractString = "Brute",
    rng::Union{AbstractRNG,Integer} = default_rng(),
    try_perserve_type::Bool = true,
)
    Xyover = tablify(
        smotenc,
        Xy,
        y_ind;
        try_perserve_type=try_perserve_type,
        encode_func = smotenc_encoder,
        decode_func = smotenc_decoder,
        k,
        ratios,
        knn_tree,
        rng,
    )
    return Xyover
end
