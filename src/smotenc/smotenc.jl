
# The following two functions are used in tablify
"""
Apply label encoding to the categorical columns of a table. A categorical column is defined
as any column with the scitype `Multiclass` or `OrderedFactor`.

# Arguments
- `X`: A table where each row is an observation which has some categorical columns
- `nominal_only::Bool`: If true, the function asserts there are only categorical columns and does
    not return their indices

# Returns
- `Xenc`: A column table where the categorical columns have been replaced by their label encoded
    versions

"""
function smotenc_encoder(X; nominal_only = false)
    # 1. Find the categorical and continuous columns
    types = ScientificTypes.schema(X).scitypes
    cat_inds = findall(x -> x <: Finite, types)
    cont_inds = findall(x -> x <: Infinite, types)
    check_scitypes_smoten_nc(length(types), cat_inds, cont_inds, types, nominal_only)

    # 2. Setup the encode and decode transforms for categotical columns
    encode_dict = Dict{Int,Function}()
    decode_dict = Dict{Int,Function}()

    columns = Tables.columns(X)
    for c in cat_inds
        column = collect(Tables.getcolumn(columns, c))
        decode_dict[c] = x -> CategoricalDistributions.decoder(column)(round(Int, x))
        encode_dict[c] = x -> CategoricalDistributions.int(x)
    end

    # 3. Encode the data
    Xenc = X |> TableOperations.transform(encode_dict) |> Tables.columntable
    # TODO: Remove Tables.columntable once https://github.com/JuliaData/TableOperations.jl/issues/32 is resolved

    # 4. SMOTE-N encoder need not pass cat_inds to tablify
    nominal_only && return Xenc, decode_dict, nothing
    return Xenc, decode_dict, cat_inds
end


"""
Decode the label encoded categorical columns of a table.

# Arguments
- `Xover`: A table where each row is an observation which has some label-encoded categorical columns
- `decode_dict`: A dictionary of functions to decode the label-encoded categorical columns

# Returns
- `Xover`: A column table where the categorical columns is decoded back to their original values
"""
function smotenc_decoder(Xover, decode_dict)
    Xover = Xover |> TableOperations.transform(decode_dict)
    return Xover
end

# SMOTE-NC uses KNN with a modified distance metric. Refer to 
# "SMOTE: Synthetic Minority Over-sampling Technique" by Chawla et al. (2002), pg. 351. 
struct EuclideanWithPenalty <: Metric
    penalty::Float64
    cont_inds::AbstractVector{<:Int}
    cat_inds::AbstractVector{<:Int}
end

"""
Given a matrix of observations `X`, find the median of the standard deviations of the
continuous features of the observations and return that as the penalty.

# Arguments
- `X::AbstractMatrix`: A matrix where each row is an observation
- `cont_inds::AbstractVector{<:Int}`: A vector of indices of the continuous features

# Returns
- `Float64`: The penalty term that modifies the distance metric

"""
function get_penalty(X::AbstractMatrix{<:AbstractFloat}, cont_inds::AbstractVector{<:Int})
    # simply compute the penalty as described
    Xcont = @view X[cont_inds, :]
    σs = vec(std(Xcont, dims = 2))
    σₘ = median(σs)
    return σₘ
end


"""
This overloads the `evaluate` function of the `Metric` struct to use the modified
distance metric which adds a penalty for each pair of corresponding categorical
variables that are not equal.

# Arguments
- `d::EuclideanWithPenalty`: The modified distance metric
- `x₁::AbstractVector`: First observation
- `x₂::AbstractVector`: Second observation

# Returns
- `Float64`: The distance between `x₁` and `x₂` using the modified distance metric
"""
function Distances.evaluate(d::EuclideanWithPenalty, x₁, x₂)
    x₁_cont, x₁_cat = x₁[d.cont_inds], x₁[d.cat_inds]
    x₂_cont, x₂_cat = x₂[d.cont_inds], x₂[d.cat_inds]
    e = euclidean(x₁_cont, x₂_cont)
    h = hamming(x₁_cat, x₂_cat)
    # distance computed as described above
    return e + d.penalty * h
end


"""
Choose a random point from the given observations matrix `X` and generate a new point that
in terms of the continuous part, randomly lies in the line joining the random point and 
randomly one of its k-nearest neighbors and in terms of the categorical part, the new point
has the mode of the categorical part of the k-nearest neighbors of the random point.

# Arguments
- `X::AbstractMatrix`: A matrix where each row is an observation
- `tree`: A k-d tree representation of the observations matrix X
- `cont_inds::AbstractVector{<:Int}`: A vector of indices of the continuous features
- `cat_inds::AbstractVector{<:Int}`: A vector of indices of the categorical features
- `k::Int`: Number of nearest neighbors to consider
- `rng::AbstractRNG`: Random number generator

# Returns
- `AbstractVector`: A new observation generated by SMOTE
"""
function generate_new_smotenc_point(
    X::AbstractMatrix{<:AbstractFloat},
    tree,
    cont_inds::AbstractVector{<:Int},
    cat_inds::AbstractVector{<:Int};
    k::Int,
    rng::AbstractRNG,
)
    # find a random point
    x_rand = randcols(rng, X)
    x_randneigh, Xneighs = get_random_neighbor(X, tree, x_rand; k, rng, return_all = true)

    # find the continuous part of new point (by choosing a random point along the line)
    x_rand_cont = @view x_rand[cont_inds]
    x_randneigh_cont = @view x_randneigh[cont_inds]
    x_new_cont = get_collinear_point(x_rand_cont, x_randneigh_cont; rng = rng)

    # find the categorical part of new point (by voting among all neighbors)
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
- `X::AbstractMatrix`: A matrix where each row is an observation
- `n::Int`: Number of new observations to generate
- `k::Int`: Number of nearest neighbors to consider.
- `cont_inds::AbstractVector{<:Int}`: A vector of indices of the continuous features
- `cat_inds::AbstractVector{<:Int}`: A vector of indices of the categorical features
- `rng::AbstractRNG`: Random number generator

# Returns
- `AbstractMatrix`: A matrix where each row is a new observation generated by SMOTE
"""
function smotenc_per_class(
    X::AbstractMatrix{<:AbstractFloat},
    n::Int,
    cont_inds::AbstractVector{<:Int},
    cat_inds::AbstractVector{<:Int};
    k::Int = 5,
    rng::AbstractRNG = default_rng(),
)
    # Can't draw lines if there are no neighbors
    n_class = size(X, 2)
    n_class == 1 && (@warn WRN_SINGLE_OBS; return X)
    # Automatically fix k if needed
    k = check_k(k, n_class)
    # Build a KNN tree with the modified distance metric
    σₘ = get_penalty(X, cont_inds)
    metric = EuclideanWithPenalty(σₘ, cont_inds, cat_inds)
    tree = BallTree(X, metric)          # May need to become BruteTree for accuracy
    # Generate n new observations
    return hcat(
        [generate_new_smotenc_point(X, tree, cont_inds, cat_inds; k, rng) for i = 1:n]...,
    )
end


"""
    function smotenc(
        X, y::AbstractVector, split_ind::Int;
        k::Int=5, ratios=nothing, rng::Union{AbstractRNG, Integer}=default_rng(),
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

- `X`: A matrix of floats or a table with scitypes that subtype `Union{Finite, Infinite}`. 
     Nominal columns should subtype `Finite` (i.e., `OrderedFactor` and `Multiclass`) and
     continuous columns should subtype `Infinite` (i.e., `Count` and `Continuous`).

- `y`: An abstract vector of labels (e.g., strings) that correspond to the observations in `X`

- `cat_inds::AbstractVector{<:Int}`: A vector of the indices of the nominal features. Supplied only if `X` is a matrix.
        Otherwise, they are inferred from the table's scitypes.


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
num_cont_feats = 3
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

julia> (Continuous, Continuous, Continuous, Continuous, Continuous)

# coerce nominal columns to a finite scitype (multiclass or ordered factor)
X = coerce(X, :Column4=>Multiclass, :Column5=>Multiclass)

# apply SMOTE-NC
Xover, yover = smotenc(X, y; k = 5, ratios = Dict(0=>1.0, 1=> 0.9, 2=>0.8), rng = 42)
StatsBase.countmap(yover)

Dict{CategoricalArrays.CategoricalValue{Int64, UInt32}, Int64} with 3 entries:
0 => 48
2 => 33
1 => 19
```
# MLJ Model Interface

Simply pass the keyword arguments while initiating the `SMOTEN` model and pass the 
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
using TableTransforms

# Generate imbalanced data
num_rows = 100
num_cont_feats = 3
y_ind = 2
# generate a table and categorical vector accordingly
Xy, _ = generate_imbalanced_data(num_rows, num_cont_feats; insert_y=y_ind,
                                probs= [0.5, 0.2, 0.3], cat_feats_num_vals=[3, 2],
                                 rng=42)  

# Table must have only finite or continuous scitypes                                
Xy = coerce(Xy, :Column2=>Multiclass, :Column5=>Multiclass, :Column6=>Multiclass)

# Initiate Random Oversampler model
oversampler = SMOTENC_t(y_ind; k=5, ratios=Dict(1=>1.0, 2=> 0.9, 3=>0.9), rng=42)
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
    k::Int = 5,
    ratios = 1.0,
    rng::Union{AbstractRNG,Integer} = default_rng(),
)
    rng = rng_handler(rng)
    # implictly infer the continuous indices
    cont_inds = setdiff(1:size(X, 2), cat_inds)
    Xover, yover =
        generic_oversample(X, y, smotenc_per_class, cont_inds, cat_inds; ratios, k, rng)
    return Xover, yover
end

# dispatch for when X is a table
function smotenc(
    X,
    y::AbstractVector;
    k::Int = 5,
    ratios = 1.0,
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
        rng,
    )
    return Xover, yover
end

# dispatch for when X is a table and y is one of its columns
function smotenc(
    Xy,
    y_ind::Int;
    k::Int = 5,
    ratios = 1.0,
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
        rng,
    )
    return Xyover
end
