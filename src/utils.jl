"""
This file contains utility functions used by the oversampling methods.
"""


# randomly sample one column of a matrix
randcols(rng::AbstractRNG, X) = X[:, rand(rng, 1:size(X, 2))]
# randomly sample n columns of a matrix
randcols(rng::AbstractRNG, X, n) = X[:, rand(rng, 1:size(X, 2), n)]
# to enable algorithms to accept either an integer or an RNG object
rng_handler(rng::Integer) = Xoshiro(rng)
rng_handler(rng::AbstractRNG) = rng


"""
Get the number of rows of a table. This implementations comes from Tables.jl as used internally there.

# Arguments
- `X`: A table

# Returns
- `Int`: Number of rows of the table
"""
function rowcount(X)
    cols = Tables.columns(X)
    names = Tables.columnnames(cols)
    isempty(names) && return 0
    return length(Tables.getcolumn(cols, names[1]))
end

"""
Return a dictionary mapping each unique value in an abstract vector to the indices of the array
where that value occurs.
"""
function group_inds(categorical_array::AbstractVector{T}) where {T}
    result = LittleDict{T,AbstractVector{Int}}()
    freeze(result)
    for (i, v) in enumerate(categorical_array)
        # Make a new entry in the dict if it doesn't exist
        if !haskey(result, v)
            result[v] = []
        end
        # It exists, so push the index belonging to the class
        push!(result[v], i)
    end
    return result
end

"""
A visual version of `StatsBase.countmap` that returns nothing. It shows how 
    many observations in the dataset belong to each class and at what proportion.

# Arguments
- `y::AbstractVector`: A vector of categorical values to test for imbalance
"""
function checkbalance(y) 
    counts = countmap(y)
    total_count = sum(values(counts))
    
    for (key, count) in counts
        percentage = round(100 * count / total_count, digits=1)
        bar_length = round(Int, count * 50 / total_count)
        bar = "▇" ^ bar_length
        println("$key: $count ($percentage%) $bar")
    end
end



"""
Generate num_rows observations of num_features features with the given probabilities of 
each class and the given type of data structure.

# Arguments
- `num_rows::Int`: Number of observations to generate
- `num_features::Int`: Number of features to generate
- `extra_cat_feats::AbstractVector`: A vector of number of levels of each extra categorical feature,
- `probs::AbstractVector`: A vector of probabilities of each class
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
    cat_feats_num_vals = [],
    probs = [0.5, 0.5],
    type = "ColTable",                
    insert_y = nothing,
    rng = default_rng(),
)
    rng = rng_handler(rng)
    # Generate y as a categorical array with classes 0, 1, 2, ..., k-1
    cum_probs = cumsum(probs)
    rands = rand(rng, num_rows)
    y = CategoricalArray([findfirst(x -> rands[i] <= x, cum_probs) - 1 for i = 1:num_rows])

    if num_cont_feats > 0
        Xc = rand(rng, Float64, num_rows, num_cont_feats)
    else
        Xc = Matrix{Int64}(undef, num_rows, 0)
    end

    for num_levels in cat_feats_num_vals
        Xc = hcat(Xc, rand(rng, 1:num_levels, num_rows))
    end

    if !isnothing(insert_y)
        Xc = hcat(Xc[:, 1:insert_y-1], y, Xc[:, insert_y:end])
    end

    DXc = Tables.table(Xc)

    if type == "Matrix"
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


