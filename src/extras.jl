
"""
    generate_imbalanced_data(num_rows, num_continuous_feats; 
                             cat_feats_num_vals=[], 
                             probs=[0.5, 0.5], 
                             insrt_y=nothing, 
                             rng=frgault_rng()
                             )

Generate `num_rows` observations with the given probabilities `probs` of 
each class. Supports generating continuous and categorical features.

# Arguments
    - `num_rows::Integer`: Number of observations to generate
    - `num_continuous_feats::Integer`: Number of features to generate
    - `cat_feats_num_vals::AbstractVector`: A vector of number of levels of each extra categorical feature.
        the number of categorical features is inferred from this.
    - `probs::AbstractVector`: A vector of probabilities of each class. The number of classes is inferred
        from this vector.
    - `insert_y::Integer`: If not nothing, insert the class labels column at the given index in the table
    - `rng::AbstractRNG`: Random number generator

# Returns
- `X:`: A column table with generated imbalanced data with `num_rows` rows and 
- `y::CategoricalArray`: An abstract vector of class labels with classes 0, 1, 2, ..., k-1
    where k is determined by the length of the probs vector

"""
function generate_imbalanced_data(
    num_rows,
    num_continuous_feats;
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
    y = CategoricalArray([findfirst(x -> rands[i] <= x, cum_probs) - 1 for i in 1:num_rows])

    if num_continuous_feats > 0
        Xc = rand(rng, Float64, num_rows, num_continuous_feats)
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

"""
    checkbalance(y; reference="majority")

A visual version of `StatsBase.countmap` that returns nothing. It prints how 
    many observations in the dataset belong to each class and their percentage
    relative to the size of majority or minority class.

# Arguments
- `y::AbstractVector`: A vector of categorical values to test for imbalance
- `reference="majority"`: Either `"majority"` or `"minority"` and decides whether the percentage should be
    relative to the size of majority or minority class.
"""
function checkbalance(y; ref = "majority")
    counts = countmap(y)
    sorted_counts = sort(collect(counts), by = x -> x[2])
    (ref in ["majority", "minority"]) || error("Invalid reference")
    ref_class_count =
        (ref == "majority") ? maximum(values(counts)) : minimum(values(counts))
    majority_class_count = maximum(values(counts))
    longest_label_length = maximum(length.(string.(keys(counts))))

    for (key, count) in sorted_counts
        percentage = round(100 * count / ref_class_count, digits = 1)
        bar_length = round(Int, count * 50 / majority_class_count)
        bar = "▇"^bar_length
        padding = " "^(longest_label_length - length(string.(key)))
        println("$key:$padding $bar $count ($percentage%) ")
    end
end
