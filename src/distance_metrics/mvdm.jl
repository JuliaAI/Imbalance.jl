
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
