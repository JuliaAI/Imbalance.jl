
"""
Labele encode each column in a given table X
"""
smoten_encoder(X) = smotenc_encoder(X; nominal_only = true)
"""
Label decode each column in a given table X
"""
smoten_decoder(X, d) = smotenc_decoder(X, d)

# SMOTE-N uses KNN with a modified distance metric. Refer to 
# "SMOTE: Synthetic Minority Over-sampling Technique" by Chawla et al. (2002), pg. 351. 

struct ValueDifference <: Metric
    # for each categorical variables with n categories, this has a nxn matrix of
    # pairwise value difference distances
    all_pairwise_vdm::AbstractVector{<:AbstractArray{<:AbstractFloat}}
end

function Distances.evaluate(d::ValueDifference, x₁, x₂)
    # Distance between two categorical vectors is the magnitude of the vector where 
    # each element is the value difference of two categories of the categorical variable
    return sum(d.all_pairwise_vdm[col][i, j]^2 for (col, (i, j)) in enumerate(zip(x₁, x₂)))
end


"""
Given a matrix of observations `X` and a vector of labels `y`, find for each column of 'X'
that has `n` categories a matrix of pairwise value difference metric distances of dimensions
`n`x`n`. The value difference metric is computed as described in the paper "SMOTE: Synthetic
Minority Over-sampling Technique" by Chawla et al. (2002), pg. 351. 

# Arguments
- `X::AbstractMatrix{<:Integer}`: A matrix of label-encoded categorical columns 
    where each row is an observation
- `y::AbstractVector`: A vector of labels

# Returns
- `AbstractVector`: A vector of pairwise value difference metric distances for each column 
    of `X`
- `AbstractVector`: A vector of the number of categories for each column of `X`
"""
function precompute_pairwise_value_difference(
    X::AbstractMatrix{<:Integer},
    y::AbstractVector,
)
    classes = unique(y)
    num_classes = length(classes)
    num_categories_per_col = [length(unique(X[:, col])) for col = 1:size(X, 2)]
    num_cols = size(X, 2)
    # function to convert a dictionary of counts to a vector of counts
    dict_to_vector = (dict, num_cats) -> [get(dict, k, 0) for k = 1:num_cats]

    # a list that maps each categorical variable (col) to a matrix that associates
    # each categorical value (per class) to its count
    all_pairwise_vdm = [
        Array{Float64}(undef, (num_classes, num_categories)) for
        num_categories in num_categories_per_col
    ]

    for col = 1:num_cols
        for (label_ind, label) in enumerate(classes)
            # count how many times each value appeared for the specific class
            all_pairwise_vdm[col][label_ind, :] =
                dict_to_vector(countmap(X[y.==label, col]), num_categories_per_col[col])
        end

        for category_ind = 1:size(all_pairwise_vdm[col], 2)
            # normalize that by the total number of times over all classes        
            normalizer = sum(all_pairwise_vdm[col][:, category_ind])
            normalizer == 0 && (all_pairwise_vdm[col][:, category_ind] .= 0; continue)
            all_pairwise_vdm[col][:, category_ind] ./= normalizer
        end

        # compute the value difference metric for all pairs of values using manhattan distance
        dist = Cityblock()
        all_pairwise_vdm[col] =
            pairwise(dist, all_pairwise_vdm[col], all_pairwise_vdm[col], dims = 2)
    end
    return all_pairwise_vdm
end


"""
Choose a random point from the given observations matrix `X` and generate a new point
by taking the mode of each categorical variable over `x` and its `k` nearest neighbors.

# Arguments
- `X::AbstractMatrix{<:Integer}`: A matrix of label-encoded categorical columns 
    where each row is an observation
- `tree`: A brute tree of `X` with a distance metric that is a `ValueDifference` object
- `k::Int`: The number of nearest neighbors to consider
- `rng::AbstractRNG`: A random number generator

# Returns
- `AbstractVector`: A vector of the mode of each categorical variable over `x` and its `k` nearest neighbors
"""
function generate_new_smoten_point(
    X::AbstractMatrix{<:Integer},
    tree;
    k::Int,
    rng::AbstractRNG,
)
    # 1. Choose a random point
    x_rand = randcols(rng, X)
    # 2. Find its k nearest neighbors (including itself)
    Xneighs = get_random_neighbor(X, tree, x_rand; k, rng, return_all_self = true)
    # 3. Find the mode of each categorical variable over the neighbors
    x_new_cat = get_neighbors_mode(Xneighs, rng)
    # 4. Return the new point
    return x_new_cat
end


"""
Assuming that all the observations in the observation matrix X belong to the same class,
use SMOTE-NC to generate `n` new observations for that class.

# Arguments
- `X::AbstractMatrix{<:Integer}`: A matrix of label-encoded categorical columns 
    where each row is an observation
- `y::AbstractVector`: A vector of labels
- `n::Int`: The number of new observations to generate
- `k::Int`: The number of nearest neighbors to consider
- `rng::AbstractRNG`: A random number generator
- `all_pairwise_vdm::AbstractVector{<:AbstractArray{<:AbstractFloat}}`: A vector of pairwise value 
   difference metric matrix for each column of `X`

# Returns
- `AbstractMatrix`: A matrix where each row is a new observation
"""
function smoten_per_class(
    X::AbstractMatrix{<:Integer},
    n::Int,
    all_pairwise_vdm::AbstractVector{<:AbstractArray{<:AbstractFloat}};
    k::Int = 5,
    rng::AbstractRNG = default_rng(),
)
    # Automatically set k to the nearest of 1 and size(X, 1) - 1
    n_class = size(X, 2)
    k = check_k(k, n_class)
    # Build KNN tree with modified distance metric
    metric = ValueDifference(all_pairwise_vdm)
    tree = BruteTree(X, metric)
    # Generate n new observations
    return hcat([generate_new_smoten_point(X, tree; k, rng) for i = 1:n]...)
end

"""
    function smoten(
        X, y::AbstractVector;
        k::Int=5, ratios=nothing, rng::Union{AbstractRNG, Integer}=default_rng(),
        try_perserve_type=true
    )

# Description
Oversamples a dataset using `SMOTE-N` (Synthetic Minority Oversampling Techniques-Nominal) algorithm to 
    correct for class imbalance as presented in [1]. This is a variant of `SMOTE` to deal with datasets 
    where all the features are nominal.


# Positional Arguments

- `X`: A matrix of integers or a table with [scitypes](https://juliaai.github.io/ScientificTypes.jl/) that subtype `Finite`. 
     That is, for table inputs each column should have either `OrderedFactor` or `Multiclass` as the element [scitype](https://juliaai.github.io/ScientificTypes.jl/).

- `y`: An abstract vector of labels (e.g., strings) that correspond to the observations in `X`

# Keyword Arguments

$DOC_COMMON_K

$DOC_RATIOS_ARGUMENT

$DOC_RNG_ARGUMENT

$DOC_TRY_PERSERVE_ARGUMENT

# Returns

$DOC_COMMON_OUTPUTS

# Example

```@repl
using Imbalance
using StatsBase

# set probability of each class
probs = [0.5, 0.2, 0.3]                         
num_rows = 100
num_cont_feats = 0
# want two categorical features with three and two possible values respectively
cat_feats_num_vals = [3, 2]

# generate a table and categorical vector accordingly
X, y = generate_imbalanced_data(num_rows, num_cont_feats; 
                                probs, cat_feats_num_vals, rng=42)                      
StatsBase.countmap(y)

julia> Dict{CategoricalArrays.CategoricalValue{Int64, UInt32}, Int64} with 3 entries:
0 => 48
2 => 33
1 => 19

ScientificTypes.schema(X).scitypes

julia> (Count, Count)

# coerce to a finite scitype (multiclass or ordered factor)
X = coerce(X, autotype(X, :few_to_finite))

# apply SMOTEN
Xover, yover = smoten(X, y; k=5, ratios=Dict(0=>1.0, 1=> 0.9, 2=>0.8), rng=42)
StatsBase.countmap(yover)


Dict{CategoricalArrays.CategoricalValue{Int64, UInt32}, Int64} with 3 entries:
0 => 48
2 => 33
1 => 19
```

# MLJ Model Interface

Simply pass the keyword arguments while initiating the `SMOTEN` model and pass the 
    positional arguments to the `transform` method. 

```julia
using MLJ
SMOTEN = @load SMOTEN pkg=Imbalance

# Wrap the model in a machine
oversampler = SMOTEN(k=5, ratios=Dict(0=>1.0, 1=> 0.9, 2=>0.8), rng=42)
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
using TableTransforms

# Generate imbalanced data
num_rows = 100
num_cont_feats = 0
y_ind = 2
# generate a table and categorical vector accordingly
Xy, _ = generate_imbalanced_data(num_rows, num_cont_feats; insert_y=y_ind,
                                probs= [0.5, 0.2, 0.3], cat_feats_num_vals=[3, 2],
                                 rng=42)  

# Table must have only finite scitypes                                
Xy = coerce(Xy, :Column1=>Multiclass, :Column2=>Multiclass, :Column3=>Multiclass)

# Initiate Random Oversampler model
oversampler = SMOTEN_t(y_ind; k=5, ratios=Dict(1=>1.0, 2=> 0.9, 3=>0.9), rng=42)
Xyover = Xy |> oversampler                               
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
    k::Int = 5,
    ratios = 1.0,
    rng::Union{AbstractRNG,Integer} = default_rng(),
)
    all_pairwise_vdm = precompute_pairwise_value_difference(X, y)
    rng = rng_handler(rng)
    Xover, yover =
        generic_oversample(X, y, smoten_per_class, all_pairwise_vdm; ratios, k, rng)
    return Xover, yover
end

# dispatch for when X is a table
function smoten(
    X,
    y::AbstractVector;
    k::Int = 5,
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
    y_ind::Int;
    k::Int = 5,
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
