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
    counts = StatsBase.countmap(y)
    total_count = sum(values(counts))
    
    for (key, count) in counts
        percentage = round(Int, 100 * count / total_count)
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



using Random

"""
Plot the data before and after oversampling in a 4x4 grid where the first row is 
the histogram of the labels before and after oversampling and the second row is the 
scatter plot of the observations before and after oversampling.

# Arguments
- `y_before::CategoricalArray`: An abstract vector of class labels before oversampling
- `y_after::CategoricalArray`: An abstract vector of class labels after oversampling
- `X_before::AbstractMatrix`: A matrix where each column is an observation before oversampling
- `X_after::AbstractMatrix`: A matrix where each column is an observation after oversampling
- `hist_only::Bool`: If true, only plot the histograms of the labels before and after oversampling. 
    If false, plot the histograms and the scatter plots of the observations before and after oversampling.
"""
function plot_data(y_before, y_after, X_before, X_after; hist_only = false)

    if Tables.istable(X_before)
        X_before = Tables.matrix(X_before)
    end

    if Tables.istable(X_after)
        X_after = Tables.matrix(X_after)
    end

    # Frequency table
    labels = unique(y_before)

    # map labels to integers
    label_map = Dict(label => i for (i, label) in enumerate(labels))

    # Find counts of each label for each version of y
    label_counts1 = [count(yi -> yi == label, y_before) for label in labels]
    label_counts2 = [count(yi -> yi == label, y_after) for label in labels]

    class_colors = distinguishable_colors(length(labels))

    # Plot the counts vs the labels in each case
    p1 = bar(
        labels,
        label_counts1,
        xlabel = "Label",
        ylabel = "Count",
        title = "\nBefore Oversampling with size $(size(X_before, 1))",
        legend = false,
    )
    p2 = bar(
        labels,
        label_counts2,
        xlabel = "Label",
        ylabel = "Count",
        title = "\nAfter Oversampling with size $(size(X_after, 1))",
        legend = false,
    )

    if !hist_only
        # Scatter plot
        p3 = scatter(
            X_before[:, 1],
            X_before[:, 2],
            xlabel = "X1",
            ylabel = "X2",
            title = "",
            legend = true,
            color = [class_colors[label_map[yi]] for yi in y_before],
        )
        p4 = scatter(
            X_after[:, 1],
            X_after[:, 2],
            xlabel = "X1",
            ylabel = "X2",
            title = "",
            legend = true,
            color = [class_colors[label_map[yi]] for yi in y_after],
        )

        plot(p1, p2, p3, p4, layout = (2, 2), size = (900, 900))
    else
        plot(p1, p2, layout = (1, 2), size = (900, 450))
    end
end
