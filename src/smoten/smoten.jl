
"""
Label encode each column in a given table X
"""
smoten_encoder(X) = generic_encoder(X; error_checker=check_scitypes_smoten)
"""
Label decode each column in a given table X
"""
smoten_decoder(X, d) = generic_decoder(X, d)

# SMOTE-N uses KNN with a modified distance metric. Refer to 
# "SMOTE: Synthetic Minority Over-sampling Technique" by Chawla et al. (2002), pg. 351. 

struct ValueDifference{T<:AbstractVector{<:AbstractArray{<:AbstractFloat}}} <: Metric
    # for each categorical variables with n categories, this has a nxn matrix of
    # pairwise value difference distances
    all_pairwise_mvdm::T
end

function Distances.evaluate(d::ValueDifference, x₁, x₂)
    # Distance between two categorical vectors is the magnitude of the vector where 
    # each element is the value difference of two categories of the categorical variable
    return sum(d.all_pairwise_mvdm[col][i, j]^2 for (col, (i, j)) in enumerate(zip(x₁, x₂)))
end
@inline (d::ValueDifference)(x₁, x₂) = Distances.evaluate(d, x₁, x₂)


"""
Given a matrix of observations `X` and a vector of labels `y`, find for each column of 'X'
that has `n` categories a matrix that associates each category with a vector of its frequencies
for each class. This is computed as described in the paper "SMOTE: Synthetic
Minority Over-sampling Technique" by Chawla et al. (2002), pg. 351. 

# Arguments
- `X`: A matrix of label-encoded categorical columns 
    where each row is an observation
- `y`: A vector of labels

# Returns
- `mvdm_encoder`: A vector of the value frequency per class matrices 
    (one for each column of `X`)
- `num_categories_per_col`: A vector of the number of categories for each column of `X`
"""
function precompute_value_encodings(
    X::AbstractMatrix{<:Integer},
    y::AbstractVector,
)
    classes = sort(unique(y))
    num_classes = length(classes)
    num_categories_per_col = [length(1:maximum(X[:, col])) for col = 1:size(X, 2)]
    num_cols = size(X, 2)
    # function to convert a dictionary of counts to a vector of counts
    dict_to_vector = (dict, num_cats) -> [get(dict, k, 0) for k = 1:num_cats]

    # a list that maps each categorical variable (col) to a matrix that associates
    # each categorical value (per class) to its count
    mvdm_encoder = [
        Array{Float64}(undef, (num_classes, num_categories)) for
        num_categories in num_categories_per_col
    ]

    for col = 1:num_cols
        for (label_ind, label) in enumerate(classes)
            # count how many times each value appeared for the specific class
            mvdm_encoder[col][label_ind, :] =
                dict_to_vector(countmap(X[y.==label, col]), num_categories_per_col[col])
        end

        for category_ind = 1:size(mvdm_encoder[col], 2)
            # normalize that by the total number of times over all classes        
            normalizer = sum(mvdm_encoder[col][:, category_ind])
            normalizer == 0 && (mvdm_encoder[col][:, category_ind] .= 0; continue)
            mvdm_encoder[col][:, category_ind] ./= normalizer
        end

    end
    return mvdm_encoder, num_categories_per_col
end

"""
Given a vector that associates each column of X to a matrix that associates each
categorical value to its frequency for every class and a vector for the number of
categories per column, this function computes the MVDM distances for each pair
of categories as described in "SMOTE: Synthetic Minority Over-sampling Technique" 
by Chawla et al. (2002), pg. 351. 

# Arguments
- `mvdm_encoder`: A vector 
    of matrices that associates each categorical value to its frequency for every
    class.
- `num_categories_per_col`: A vector of integers
    representing the number of categories per column.

# Returns
- `mvdm_distances`: A vector
    of matrices that associates with each column in X a matrix that stores the
    mvdm distance component between any two categories.
"""
function precompute_mvdm_distances(
    mvdm_encoder::AbstractVector{<:AbstractArray{<:AbstractFloat}},
    num_categories_per_col::AbstractVector{<:Integer},
)
    num_cols = length(mvdm_encoder) 
    all_pairwise_mvdm = [
        Array{Float64}(undef, (num_categories, num_categories)) for
        num_categories in num_categories_per_col
    ]
    for col = 1:num_cols
        dist = Cityblock()
        all_pairwise_mvdm[col] = pairwise(dist, mvdm_encoder[col],  dims = 2).^2
    end
    return all_pairwise_mvdm
end


"""
Choose a random point from the given observations matrix `X` and generate a new point
by taking the mode of each categorical variable over `x` and its `k` nearest neighbors.

# Arguments
- `X`: A matrix of label-encoded categorical columns 
    where each row is an observation
- `tree`: A brute tree of `X` with a distance metric that is a `ValueDifference` object
- `k`: The number of nearest neighbors to consider
- `rng`: A random number generator

# Returns
- `x_new_cat`: A vector of the mode of each categorical variable over `x` and its `k` nearest neighbors
"""
function generate_new_smoten_point(
    X::AbstractMatrix{<:Integer},
    knn_matrix;
    k::Integer,
    rng::AbstractRNG,
)
    # 1. Choose a random point (by index)
    ind = rand(rng, 1:size(X, 2))
    # 2. Find its k nearest neighbors (including itself)
    Xneighs = X[:, knn_matrix[ind]]
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
- `y`: A vector of labels
- `n`: The number of new observations to generate
- `k`: The number of nearest neighbors to consider
- `rng`: A random number generator
- `all_pairwise_mvdm`: A vector of pairwise value 
   difference metric matrix for each column of `X`

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
    knn_matrix, _ = knn(tree, X, k + 1)

    # Generate n new observations
     Xnew = zeros(Float32, size(X, 1), n)
     p = Progress(n)
     for i=1:n
         Xnew[:, i] = generate_new_smoten_point(X, knn_matrix; k, rng)
         next!(p)
     end
    return Xnew
end

"""
    smoten(
        X, y;
        k, ratios=nothing, rng=default_rng(),
        try_perserve_type=true
    )

# Description
Oversamples a dataset using `SMOTE-N` (Synthetic Minority Oversampling Techniques-Nominal) algorithm to 
    correct for class imbalance as presented in [1]. This is a variant of `SMOTE` to deal with datasets 
    where all the features are nominal.


# Positional Arguments

- `X`: A matrix of integers or a table with element[scitypes](https://juliaai.github.io/ScientificTypes.jl/) that subtype `Finite`. 
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
using StatsBase

# set probability of each class
probs = [0.5, 0.2, 0.3]                         
num_rows = 100
num_continuous_feats = 0
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
(Count, Count)

# coerce to a finite scitype (multiclass or ordered factor)
X = coerce(X, autotype(X, :few_to_finite))

# apply SMOTEN
Xover, yover = smoten(X, y; k=5, ratios=Dict(0=>1.0, 1=> 0.9, 2=>0.8), rng=42)
julia> StatsBase.countmap(yover)
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
using Imbalance.TableTransforms

# Generate imbalanced data
num_rows = 100
num_continuous_feats = 0
y_ind = 2
# generate a table and categorical vector accordingly
Xy, _ = generate_imbalanced_data(num_rows, num_continuous_feats; insert_y=y_ind,
                                probs= [0.5, 0.2, 0.3], cat_feats_num_vals=[3, 2],
                                 rng=42)  

# Table must have only finite scitypes                                
Xy = coerce(Xy, :Column1=>Multiclass, :Column2=>Multiclass, :Column3=>Multiclass)

# Initiate Random Oversampler model
oversampler = SMOTEN(y_ind; k=5, ratios=Dict(1=>1.0, 2=> 0.9, 3=>0.9), rng=42)
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
    k::Integer = 5,
    ratios = 1.0,
    rng::Union{AbstractRNG,Integer} = default_rng(),
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
