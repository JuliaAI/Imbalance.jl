# SMOTE-N uses KNN with a modified distance metric. Refer to 
# "SMOTE: Synthetic Minority Over-sampling Technique" by Chawla et al. (2002), pg. 351. 

struct ValueDifference <: Metric
    all_pairwise_vdm::AbstractVector{<:AbstractArray{<:AbstractFloat}}
end

function Distances.evaluate(d::ValueDifference, x₁, x₂)
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
function precompute_pairwise_value_difference(X::AbstractMatrix{<:Integer}, y::AbstractVector)
    classes = unique(y)
    num_classes = length(classes)
    categories_per_col = [length(unique(X[:, col])) for col in 1:size(X, 2)]
    num_cols = size(X, 2)
    dict_to_vector = (dict, num_cats) -> [get(dict, k, 0) for k in 1:num_cats]

    # a list that maps each categorical variable (col) to a matrix that associates
    # each categorical value (per class) to its count
    all_pairwise_vdm = [Array{Float64}(undef, (num_classes, num_categories)) 
                  for num_categories in categories_per_col]

    for col in 1:num_cols
        for (label_ind, label) in enumerate(classes)
            all_pairwise_vdm[col][label_ind, :] = dict_to_vector(countmap(X[y .== label, col]), categories_per_col[col])
        end

        for category_ind in 1:size(all_pairwise_vdm[col], 2)
            all_pairwise_vdm[col][:, category_ind] ./= sum(all_pairwise_vdm[col][:, category_ind])
        end

        # 0/0 may be possible
        all_pairwise_vdm[col][isnan.(all_pairwise_vdm[col])] .= 0

        dist = Cityblock()
        all_pairwise_vdm[col] = pairwise(dist, all_pairwise_vdm[col], all_pairwise_vdm[col])
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
    x_rand = randcols(rng, X)
    Xneighs = get_random_neighbor(X, tree, x_rand; k, rng, return_all_self = true)
    x_new_cat = get_neighbors_mode(Xneighs, rng)
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
    size(X, 2) == 1 && (@warn "class with a single observation will be ignored"; return X)
    k = (k > 0) ? min(k, size(X, 1) - 1) : 1
    metric = ValueDifference(all_pairwise_vdm)
    tree = BruteTree(X, metric)
    return hcat([generate_new_smoten_point(X, tree; k, rng) for i = 1:n]...)
end

"""
function smoten(
    X, y::AbstractVector;
    k::Int=5, ratios=nothing, rng::Union{AbstractRNG, Integer}=default_rng()
)

Oversample a dataset given by a matrix or table of observations `X` and an abstract vector of labels y using SMOTE-N.

# Arguments

$DOC_COMMON_INPUTS

- `k::Int`: Number of nearest neighbors to consider in the SMOTE algorithm. 
Should be within the range `[1, size(X, 1) - 1]` else set to the nearest of these two values.

$DOC_RATIOS_ARGUMENT

$DOC_RNG_ARGUMENT

# Returns

$DOC_COMMON_OUTPUTS
"""
function smoten(
    X::AbstractMatrix{<:Integer},
    y::AbstractVector;
    k::Int = 5,
    ratios = nothing,
    rng::Union{AbstractRNG,Integer} = default_rng(),
)
    all_pairwise_vdm = precompute_pairwise_value_difference(X, y)
    rng = rng_handler(rng)
    Xover, yover =
        generic_oversample(X, y, smoten_per_class, all_pairwise_vdm; ratios, k, rng )
    return Xover, yover
end

function smoten(
    X,
    y::AbstractVector;
    k::Int = 5,
    ratios = nothing,
    rng::Union{AbstractRNG,Integer} = default_rng(),
)
    Xover, yover = tablify(smoten, X, y; encode_func=smoten_encoder, decode_func=smoten_decoder, materialize=true, k, ratios, rng)
    return Xover, yover
end

function smoten(
    Xy,
    y_ind::Int;
    k::Int = 5,
    ratios = nothing,
    rng::Union{AbstractRNG,Integer} = default_rng(),
)
    Xyover = tablify(smoten, Xy, y_ind; encode_func=smoten_encoder, decode_func=smoten_decoder, materialize=true, k, ratios, rng)
    return Xyover
end

smoten_encoder(X) = smotenc_encoder(X; nominal_only=true)
smoten_decoder(X, d) = smotenc_decoder(X, d)
