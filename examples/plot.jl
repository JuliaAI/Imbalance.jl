import Pkg; Pkg.add("Colors")
using Colors

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
            legend = false,
            color = [class_colors[label_map[yi]] for yi in y_before],
        )
        p4 = scatter(
            X_after[:, 1],
            X_after[:, 2],
            xlabel = "X1",
            ylabel = "X2",
            title = "",
            legend = false,
            color = [class_colors[label_map[yi]] for yi in y_after],
            markershape = :diamond
        )

        plot(p1, p2, p3, p4, layout = (2, 2), size = (900, 900))
    else
        plot(p1, p2, layout = (1, 2), size = (900, 650))
    end
end
