
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
    rng = Imbalance.rng_handler(rng)
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
    # Find labels of y
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
