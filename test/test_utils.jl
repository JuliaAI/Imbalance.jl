

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
    # Get the number of classes from the length of the probabilities vector
    num_classes = length(probs)
    # Generate y as a categorical array with classes 0, 1, 2, ..., k-1
    cum_probs = cumsum(probs)
    rands = rand(rng, num_rows)
    y = CategoricalArray([findfirst(x -> rands[i] <= x , cumsum(probs)) - 1 for i in 1:num_rows])
    return X, y
end



function plot_data(y_before, y_after, X_before, X_after)

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
    
    class_colors = [:blue, :yellow, :red]

    # Plot the counts vs the labels in each case
    p1 = bar(labels, label_counts1, xlabel="Label", ylabel="Count", title="\nBefore Oversampling", legend=false)
    p2 = bar(labels, label_counts2, xlabel="Label", ylabel="Count", title="\nAfter Oversampling", legend=false)
    
    # Scatter plot
    p3 = scatter(X_before[:, 1], X_before[:, 2], xlabel="X1", ylabel="X2", title="Before Oversampling with size $(size(X_before)[1])", legend=false,
                color=[class_colors[label_map[yi]] for yi in y_before])
    p4 = scatter(X_after[:, 1], X_after[:, 2], xlabel="X1", ylabel="X2", title="After Oversampling with size $(size(X_after)[1])", legend=false,
                color=[class_colors[label_map[yi]] for yi in y_after])
    
    # Plotting the figures together in a 2x2 layout
    plot(p1, p2, p3, p4,   layout=(2, 2), size=(900, 900))
end
