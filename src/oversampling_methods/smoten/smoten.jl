
# SMOTE-N uses KNN with a modified distance metric. Refer to 
# "SMOTE: Synthetic Minority Over-sampling Technique" by Chawla et al. (2002), pg. 351. 
include("../../distance_metrics/mvdm.jl")

"""
Label encoding and decoding
"""
smoten_encoder(X) = generic_encoder(X; error_checker=check_scitypes_smoten)
smoten_decoder(X, d) = generic_decoder(X, d)


"""
Check that all columns are either categorical or continuous. If not, throw an error.

# Arguments
- `ncols`: Number of columns
- `cat_inds`: Indices of categorical columns
- `cont_inds`: Indices of continuous columns
- `types`: Types of each column

"""
function check_scitypes_smoten(ncols, cat_inds, cont_inds, types)
    bad_cols = setdiff(1:ncols, cat_inds)
    if !isempty(bad_cols)
        throw(ArgumentError(ERR_BAD_NOM_COL_TYPES(bad_cols, types[bad_cols])))
    end
    return
end

# Used by SMOTENC and SMOTEN
"""
Find the mode of each row in a matrix and return the result as a vector. If multiple ⊧
exist, choose one of them randomly.

# Arguments
- `Xneights`: A matrix where each row is an observation of real numbers
- `rng`: Random number generator

# Returns
-  A vector where each element is the mode of the corresponding row in `A`
"""
function get_neighbors_mode(
    Xneighs::AbstractMatrix{<:Real},
    rng::AbstractRNG = default_rng(),
)
    # fair voting by taking the mode over each nominal variable (row)
    # shuffle to avoid bias when breaking ties
    return [mode(shuffle(rng, row)) for row in eachrow(Xneighs)]
end

"""
Choose a random point from the given observations matrix `X` and generate a new point
by taking the mode of each categorical variable over `x` and its `k` nearest neighbors.

# Arguments
- `X`: A matrix of label-encoded categorical columns 
    where each row is an observation
- `knn_map`: A vector of vectors that maps each points to its neighbors
- `rng`: A random number generator

# Returns
- `x_new_cat`: A vector of the mode of each categorical variable over `x` and its `k` nearest neighbors
"""
function generate_new_smoten_point(
    X::AbstractMatrix{<:Integer},
    knn_map;
    rng::AbstractRNG,
)
    # 1. Choose a random point (by index)
    ind = rand(rng, 1:size(X, 2))
    # 2. Find its k nearest neighbors (including itself)
    Xneighs = X[:, knn_map[ind]]
    # 3. Find the mode of each categorical variable over the neighbors
    x_new_cat = get_neighbors_mode(Xneighs, rng)
    # 4. Return the new point
    return x_new_cat
end


"""
Assuming that all the observations in the observation matrix X belong to the same class,
use SMOTE-NC to generate `n` new observations for that class.

# Arguments
- `X`: A matrix of label-encoded categorical columns 
    where each row is an observation
- `n`: The number of new observations to generate
- `all_pairwise_mvdm`: A vector of pairwise value 
   difference metric matrix for each column of `X`
- `k`: The number of nearest neighbors to consider
- `rng`: A random number generator

# Returns
- `Xnew`: A matrix where each row is a new observation
"""
function smoten_per_class(
    X::AbstractMatrix{<:Real},
    n::Integer,
    all_pairwise_mvdm::AbstractVector{<:AbstractArray{<:AbstractFloat}};
    k::Integer = 5,
    rng::AbstractRNG = default_rng(),
)
    X = Int32.(X)               # temporary workaround for an unexepcted types bug

    # Automatically set k to the nearest of 1 and size(X, 1) - 1
    n_class = size(X, 2)
    k = check_k(k, n_class)
    
    # Build KNN tree with modified distance metric
    metric = ValueDifference(all_pairwise_mvdm)
    tree = BruteTree(X, metric)
    knn_map, _ = knn(tree, X, k + 1)

    # Generate n new observations
     Xnew = zeros(Float32, size(X, 1), n)
     p = Progress(n)
     for i=1:n
         Xnew[:, i] = generate_new_smoten_point(X, knn_map; rng)
         next!(p)
     end
    return Xnew
end

"""
    smoten(
        X, y;
        k=5, ratios=1.0, rng=default_rng(),
        try_perserve_type=true
    )

# Description
Oversamples a dataset using `SMOTE-N` (Synthetic Minority Oversampling Techniques-Nominal) algorithm to 
    correct for class imbalance as presented in [1]. This is a variant of `SMOTE` to deal with datasets 
    where all features are nominal.


# Positional Arguments

- `X`: A matrix of integers or a table with element [scitypes](https://juliaai.github.io/ScientificTypes.jl/) that subtype `Finite`. 
     That is, for table inputs each column should have either `OrderedFactor` or `Multiclass` as the element [scitype](https://juliaai.github.io/ScientificTypes.jl/).

- `y`: An abstract vector of labels (e.g., strings) that correspond to the observations in `X`

# Keyword Arguments

$(COMMON_DOCS["K"])

$(COMMON_DOCS["RATIOS"])

$(COMMON_DOCS["RNG"])

$(COMMON_DOCS["TRY_PERSERVE_TYPE"])

# Returns

$(COMMON_DOCS["OUTPUTS"])

# Example

```@repl
using Imbalance

# set probability of each class
class_probs = [0.5, 0.2, 0.3]                         
num_rows = 100
num_continuous_feats = 0
# want two categorical features with three and two possible values respectively
num_vals_per_category = [3, 2]

# generate a table and categorical vector accordingly
X, y = generate_imbalanced_data(num_rows, num_continuous_feats; 
                                class_probs, num_vals_per_category, rng=42)                      
julia> Imbalance.countmap(y)
1: ▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇ 19 (39.6%) 
2: ▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇ 33 (68.8%) 
0: ▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇ 48 (100.0%) 

julia> ScientificTypes.schema(X).scitypes
(Count, Count)

# coerce to a finite scitype (multiclass or ordered factor)
X = coerce(X, autotype(X, :few_to_finite))

# apply SMOTEN
Xover, yover = smoten(X, y; k=5, ratios=Dict(0=>1.0, 1=> 0.9, 2=>0.8), rng=42)

julia> Imbalance.countmap(y)
2: ▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇ 38 (79.2%) 
1: ▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇ 43 (89.6%) 
0: ▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇ 48 (100.0%) 
```

# MLJ Model Interface

Simply pass the keyword arguments while initiating the `SMOTEN` model and pass the 
    positional arguments `X, y` to the `transform` method. 

```julia
using MLJ
SMOTEN = @load SMOTEN pkg=Imbalance

# Wrap the model in a machine
oversampler = SMOTEN(k=5, ratios=Dict(0=>1.0, 1=> 0.9, 2=>0.8), rng=42)
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
using ScientificTypes
using Imbalance.TableTransforms

# Generate imbalanced data
num_rows = 100
num_continuous_feats = 0
y_ind = 2
# generate a table and categorical vector accordingly
Xy, _ = generate_imbalanced_data(num_rows, num_continuous_feats; insert_y=y_ind,
                                class_probs= [0.5, 0.2, 0.3], num_vals_per_category=[3, 2],
                                 rng=42)  

# Table must have only finite scitypes                                
Xy = coerce(Xy, :Column1=>Multiclass, :Column2=>Multiclass, :Column3=>Multiclass)

# Initiate SMOTEN model
oversampler = SMOTEN(y_ind; k=5, ratios=Dict(1=>1.0, 2=> 0.9, 3=>0.9), rng=42)
Xyover = Xy |> oversampler                               
# equivalently if TableTransforms is used
Xyover, cache = TableTransforms.apply(oversampler, Xy)    # equivalently
```
The `reapply(oversampler, Xy, cache)` method from `TableTransforms` simply falls back to `apply(oversample, Xy)` and the `revert(oversampler, Xy, cache)`
reverts the transform by removing the oversampled observations from the table.


# References
[1] N. V. Chawla, K. W. Bowyer, L. O.Hall, W. P. Kegelmeyer,
“SMOTE: synthetic minority over-sampling technique,”
Journal of artificial intelligence research, 321-357, 2002.
"""
function smoten(
    X::AbstractMatrix{<:Integer},
    y::AbstractVector;
    k::Integer = 5,
    ratios = 1.0,
    rng::Union{AbstractRNG,Integer} = default_rng(),
    try_perserve_type::Bool = true,
)
    mvdm_encoder, num_categories_per_col = precompute_value_encodings(X, y)
    all_pairwise_mvdm = precompute_mvdm_distances(mvdm_encoder, num_categories_per_col)
    rng = rng_handler(rng)
    Xover, yover =
        generic_oversample(X, y, smoten_per_class, all_pairwise_mvdm; ratios, k, rng)
    return Xover, yover
end

# dispatch for when X is a table
function smoten(
    X,
    y::AbstractVector;
    k::Integer = 5,
    ratios = 1.0,
    rng::Union{AbstractRNG,Integer} = default_rng(),
    try_perserve_type::Bool = true,
)
    Xover, yover = tablify(
        smoten,
        X,
        y;
        try_perserve_type=try_perserve_type,
        encode_func = smoten_encoder,
        decode_func = smoten_decoder,
        k,
        ratios,
        rng,
    )
    return Xover, yover
end

# dispatch for when X is a table and y is one of the columns
function smoten(
    Xy,
    y_ind::Integer;
    k::Integer = 5,
    ratios = 1.0,
    rng::Union{AbstractRNG,Integer} = default_rng(),
    try_perserve_type::Bool = true,
)
    Xyover = tablify(
        smoten,
        Xy,
        y_ind;
        try_perserve_type=try_perserve_type,
        encode_func = smoten_encoder,
        decode_func = smoten_decoder,
        k,
        ratios,
        rng,
    )
    return Xyover
end
