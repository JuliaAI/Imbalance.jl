### Contains functions to be used for testing purposes
rng = StableRNG(123)

"""
Generate num_rows observations of num_features features with the given probabilities of 
each class and the given type of data structure.

# Arguments
- `num_rows::Int`: Number of observations to generate
- `num_features::Int`: Number of features to generate
- `extra_cat_feats::AbstractVector`: A vector of number of levels of each extra categorical feature,
- `probs::AbstractVector`: A vector of probabilities of each class
- `type::String`: Type of data structure to generate. Valid values are "DF", 
    "Matrix", "RowTable", "ColTable", "MatrixTable", "DictRowTable", "DictColTable"
- `insert_y::Int`: If not nothing, insert the class label at the given index
- `rng::AbstractRNG`: Random number generator

# Returns
- `X:`: A table or matrix where each row is an observation of floats
- `y::CategoricalArray`: An abstract vector of class labels with classes 0, 1, 2, ..., k-1
    where k is determined by the length of the probs vector
"""

function generate_imbalanced_data(
    num_rows,
    num_cont_feats;
    extra_cat_feats = [],
    probs = [0.5, 0.5],
    type = "DF",
    insert_y = nothing,
    rng = Random.default_rng(),
)
    rng = rng_handler(rng)
    # Generate y as a categorical array with classes 0, 1, 2, ..., k-1
    cum_probs = cumsum(probs)
    rands = rand(rng, num_rows)
    y = CategoricalArray([findfirst(x -> rands[i] <= x, cum_probs) - 1 for i = 1:num_rows])

    Xc = rand(rng, Float64, num_rows, num_cont_feats)
    for num_levels in extra_cat_feats
        Xc = hcat(Xc, rand(rng, 1:num_levels, num_rows))
    end

    if !isnothing(insert_y)
        Xc = hcat(Xc[:, 1:insert_y-1], y, Xc[:, insert_y:end])
    end

    DXc = DataFrame(Xc, :auto)

    if type == "DF"
        X = DXc
    elseif type == "Matrix"
        X = Xc
    elseif type == "RowTable"
        X = Tables.rowtable(DXc)
    elseif type == "ColTable"
        X = Tables.columntable(DXc)
    elseif type == "MatrixTable"
        X = Tables.table(Xc)
    elseif type == "DictRowTable"
        X = Tables.dictrowtable(DXc)
    elseif type == "DictColTable"
        X = Tables.dictcolumntable(DXc)
    else
        error("Invalid type")
    end

    return X, y
end



"""
Test if point a is between points b and c

# Arguments
- `a::AbstractVector`: A point
- `b::AbstractVector`: A point
- `c::AbstractVector`: A point

# Returns
- `Bool`: True if a is between b and c, false otherwise
"""
function is_in_between(a, b, c; atol = 0.01)::Bool
    dist_ab = sqrt(sum((a .- b) .^ 2))
    dist_ac = sqrt(sum((a .- c) .^ 2))
    dist_total = sqrt(sum((b .- c) .^ 2))
    return isapprox(dist_ab + dist_ac, dist_total; atol)
end
