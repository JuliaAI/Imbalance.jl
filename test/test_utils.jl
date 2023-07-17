### Contains functions to be used for testing purposes


rng = Random.default_rng(123)

"""
Generate num_rows observations of num_features features with the given probabilities of each class and the given type of data structure.

# Arguments
- `num_rows::Int`: Number of observations to generate
- `num_features::Int`: Number of features to generate
- `probs::AbstractVector`: A vector of probabilities of each class
- `type::String`: Type of data structure to generate. Valid values are "DF", "Matrix", "RowTable", "ColTable", "MatrixTable", "DictRowTable", "DictColTable"
- `rng::AbstractRNG`: Random number generator

# Returns
- `X:`: A table or matrix where each row is an observation of floats
- `y::CategoricalArray`: A categorical vector of labels with classes 0, 1, 2, ..., k-1 where k is determined by the length of the probs vector

"""
function generate_imbalanced_data(num_rows, num_features; probs=[0.5, 0.5], type="DF", rng=Random.default_rng())
    if type == "DF"
        X = DataFrame(rand(rng, Float64, num_rows, num_features), :auto)
    elseif type == "Matrix"
        X = rand(rng, Float64, num_rows, num_features)
    elseif type == "RowTable"
        X = DataFrame(rand(rng, Float64, num_rows, num_features), :auto)
        X = Tables.rowtable(X)
    elseif type == "ColTable"
        X = DataFrame(rand(rng, Float64, num_rows, num_features), :auto)
        X = Tables.columntable(X)
    elseif type == "MatrixTable"
        X = rand(rng, Float64, num_rows, num_features)
        X = Tables.table(X)
    elseif type == "DictRowTable"
        X = DataFrame(rand(rng, Float64, num_rows, num_features), :auto)
        X = Tables.dictrowtable(X)
    elseif type == "DictColTable"
        X = DataFrame(rand(rng, Float64, num_rows, num_features), :auto)
        X = Tables.dictcolumntable(X)
    else 
        error("Invalid type")
    end
    # Generate y as a categorical array with classes 0, 1, 2, ..., k-1
    cum_probs = cumsum(probs)
    rands = rand(rng, num_rows)
    y = CategoricalArray([findfirst(x -> rands[i] <= x , cum_probs) - 1 for i in 1:num_rows])
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
function _is_in_between(a, b, c; atol=0.01)::Bool
    dist_ab = sqrt(sum((a .- b).^2))
    dist_ac = sqrt(sum((a .- c).^2))
    dist_total = sqrt(sum((b .- c).^2))
    return isapprox(dist_ab + dist_ac, dist_total; atol)
end