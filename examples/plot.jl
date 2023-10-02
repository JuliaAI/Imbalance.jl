import Pkg; Pkg.add("Colors")
using Colors

"""
Plot the data before and after resampling in a 4x4 grid where the first row is 
	the histogram of the labels before and after oversampling and the second row is the 
	scatter plot of the observations before and after oversampling. Oversampled points are
	indicated with a diamond and undersampled points are indicated with a cross.

# Arguments
- `y_before::CategoricalArray`: An abstract vector of class labels before resampling
- `y_after::CategoricalArray`: An abstract vector of class labels after resampling
- `X_before::AbstractMatrix`: A matrix where each column is an observation before resampling
- `X_after::AbstractMatrix`: A matrix where each column is an observation after resampling
- `points_only::Bool`: If true, only plot the points before and after resampling
"""
function plot_data(
	y_before,
	y_after,
	X_before,
	X_after;
	points_only = false,
	show_deleted = true,
)
	# if table then convert to matrix
	Tables.istable(X_before) && (X_before = Tables.matrix(X_before))
	Tables.istable(X_after) && (X_after = Tables.matrix(X_after))

	# decide whether data has been oversampled or undersampled
	num_new_rows = size(X_after, 1) - size(X_before, 1)
	mode = (num_new_rows >= 0) ? "oversampling" : "undersampling"

	# a function to get the oversampled or undersampled points only
	# exploit duality in terms of set difference in both
	X_new = Array{Float64}(undef, size(X_before, 2), 0)
	y_new::Vector{Int} = []
	function get_new_points(X_ref, y_ref, X_check)
		# function that finds points and labels in X_ref but not X_check
		for i in 1:size(X_ref, 1)
			row = @view X_ref[i, :]
			if row ∉ eachrow(X_check)
				X_new = hcat(X_new, X_ref[i, :])
				push!(y_new, y_ref[i])
			end
		end
		return X_new', y_new
	end

	# use the function to get the new points 
	# (i.e., added by oversampling or removed by undersampling)
	if mode == "oversampling"
		X_ref, y_ref = X_after, y_after
		X_check, y_check = X_before, y_before
	else
		X_ref, y_ref = X_before, y_before
		X_check, y_check = X_after, y_after
	end
	X_new, y_new = get_new_points(X_ref, y_ref, X_check)
	old_count = size(X_before, 1)

	# in case it's random oversampling no new points are added...
	if length(y_new) == 0
		X_new = X_after[old_count:end, :]
		y_new = y_after[old_count:end]
	end

	# Frequency table
	labels = unique(y_before)
	# map labels to integers
	label_map = Dict(label => i for (i, label) in enumerate(labels))
	class_colors =
		distinguishable_colors(length(labels), [RGB(1, 0, 0), RGB(0, 1, 0), RGB(0, 0, 1)])

	# Scatter plot
	p3 = plot()
	[
		scatter!(p3, X_before[:, 1][y_before.==y], X_before[:, 2][y_before.==y],
			xlabel = "\$x_{1}\$",
			ylabel = "\$x_{2}\$", label = "\$y=$y\$", color = class_colors[label_map[y]],
			markerstrokewidth = 0.5,
			legend_font_pointsize = 7,
		) for y in unique(y_before)
	]

	p4 = plot()
	[
		scatter!(p4, X_check[:, 1][y_check.==y], X_check[:, 2][y_check.==y],
			xlabel = "\$x_{1}\$",
			ylabel = "\$x_{2}\$", label = "\$y=$y\$", color = class_colors[label_map[y]],
			markerstrokewidth = 0.5,
			legend_font_pointsize = 7,
		) for y in unique(y_check)
	]

	if (mode == "undersampling" && show_deleted) || mode == "oversampling"

		[
			scatter!(p4, X_new[:, 1][y_new.==y], X_new[:, 2][y_new.==y],
				xlabel = "\$x_{1}\$",
				ylabel = "\$x_{2}\$",
				label = (mode == "oversampling") ? "\$y=$y\\:(added)\$" :
						"\$y=$y\\:(deleted)\$",
				markershape = (mode == "oversampling") ? :diamond : :xcross,
				markersize = (mode == "oversampling") ? 4 : 2.5,
				color = class_colors[label_map[y]],
				markerstrokewidth = 0.5,
				legend_font_pointsize = 7,
			) for y in unique(y_new)
		]
	end

	if !points_only
		# Find counts of each label for each version of y
		label_counts_before = [count(yi -> yi == label, y_before) for label in labels]
		label_counts_after = [count(yi -> yi == label, y_after) for label in labels]

		# Plot the counts vs the labels in each case
		p1 = bar(
			labels,
			label_counts_before,
			xlabel = "Label",
			ylabel = "Count",
			legend = false,
		)
		p2 = bar(
			labels,
			label_counts_after,
			xlabel = "Label",
			ylabel = "Count",
			legend = false,
		)
		plot(
			p1,
			p2,
			p3,
			p4,
			layout = (2, 2),
			size = (900, 900),
			plot_title = "Data before and after $mode",
		)
	else
		plot(
			p3,
			p4,
			layout = (1, 2),
			size = (900, 450),
			plot_title = "Data before and after $mode",
		)
	end

end