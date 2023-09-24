# Imports


```julia
using Random
using CSV
using DataFrames
using MLJ
using ScientificTypes
using Imbalance
using Plots
```

## Loading Data

Let's load the Iris dataset, the objective of this dataset is to predict the type of flower as one of "virginica", "versicolor" and "setosa" using its sepal and petal length and width.

We don't need to so from a CSV file this time because `MLJ` has a macro for loading it already! The only difference is that we will need to explictly convert it to a dataframe as `MLJ` loads it as a named tuple of vectors.


```julia
X, y = @load_iris
X = DataFrame(X)
first(X, 5) |> pretty
```

    ┌──────────────┬─────────────┬──────────────┬─────────────┐
    │ sepal_length │ sepal_width │ petal_length │ petal_width │
    │ Float64      │ Float64     │ Float64      │ Float64     │
    │ Continuous   │ Continuous  │ Continuous   │ Continuous  │
    ├──────────────┼─────────────┼──────────────┼─────────────┤
    │ 5.1          │ 3.5         │ 1.4          │ 0.2         │
    │ 4.9          │ 3.0         │ 1.4          │ 0.2         │
    │ 4.7          │ 3.2         │ 1.3          │ 0.2         │
    │ 4.6          │ 3.1         │ 1.5          │ 0.2         │
    │ 5.0          │ 3.6         │ 1.4          │ 0.2         │
    └──────────────┴─────────────┴──────────────┴─────────────┘


Our purpose for this tutorial is primarily visuallization. Thus, let's select two of the continuous features only to work with. It's known that the sepal length and width play a much bigger role in classifying the type of flower so let's keep those only.


```julia
X = select(X, :petal_width, :petal_length)
first(X, 5) |> pretty
```

    ┌─────────────┬──────────────┐
    │ petal_width │ petal_length │
    │ Float64     │ Float64      │
    │ Continuous  │ Continuous   │
    ├─────────────┼──────────────┤
    │ 0.2         │ 1.4          │
    │ 0.2         │ 1.4          │
    │ 0.2         │ 1.3          │
    │ 0.2         │ 1.5          │
    │ 0.2         │ 1.4          │
    └─────────────┴──────────────┘


## Coercing Data


```julia
ScientificTypes.schema(X)
```


    ┌──────────────┬────────────┬─────────┐
    │ names        │ scitypes   │ types   │
    ├──────────────┼────────────┼─────────┤
    │ petal_width  │ Continuous │ Float64 │
    │ petal_length │ Continuous │ Float64 │
    └──────────────┴────────────┴─────────┘



Things look good, no coercion is needed.


## Oversampling

Iris, by default has no imbalance problem


```julia
checkbalance(y)
```

    virginica:  ▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇ 50 (100.0%) 
    setosa:     ▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇ 50 (100.0%) 
    versicolor: ▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇ 50 (100.0%) 


To simulate that there is a balance problem, we will consider a random sample of 100 observations. A random sample does not guarantee perserving the proportion of classes; in this, we actually set the seed to get a very unlikely random sample that suffers from moderate imbalance.


```julia
Random.seed!(803429)
subset_indices = rand(1:size(X, 1), 100)
X, y = X[subset_indices, :], y[subset_indices]
checkbalance(y)         # comes from Imbalance
```

    versicolor: ▇▇▇▇▇▇▇▇▇▇▇ 12 (22.6%) 
    setosa:     ▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇ 35 (66.0%) 
    virginica:  ▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇ 53 (100.0%) 


We will treat this as our training set going forward so we don't need to partition. Now let's oversample it with SMOTE.


```julia
Xover, yover = smote(X, y; k=5, ratios=Dict("versicolor" => 0.7), rng=42)
checkbalance(yover)
```

    setosa:     ▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇ 35 (66.0%) 
    versicolor: ▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇ 37 (69.8%) 
    virginica:  ▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇ 53 (100.0%) 


## Training the Model


```julia
models(matching(Xover, yover))
```

Let's go for an SVM


```julia
import Pkg;
Pkg.add("MLJLIBSVMInterface");
```

### Before Oversampling


```julia
# 1. Load the model
SVC = @load SVC pkg = LIBSVM

# 2. Instantiate it (γ=0.01 is intentional)
model = SVC(gamma=0.01)

# 3. Wrap it with the data in a machine
mach = machine(model, X, y)

# 4. fit the machine learning model
fit!(mach)
```

### After Oversampling


```julia
# 3. Wrap it with the data in a machine
mach_over = machine(model, Xover, yover)

# 4. fit the machine learning model
fit!(mach_over)
```

## Plot Decision Boundaries

Construct ranges for each feature and consecutively a grid


```julia
petal_width_range =
	range(minimum(X.petal_width) - 1, maximum(X.petal_width) + 1, length = 200)
petal_length_range =
	range(minimum(X.petal_length) - 1, maximum(X.petal_length) + 1, length = 200)
grid_points = [(pw, pl) for pw in petal_width_range, pl in petal_length_range]
```

Evaluate the grid with the machine before and after oversampling


```julia
grid_predictions =[
    predict(mach, Tables.table(reshape(collect(point), 1, 2)))[1] for
 	point in grid_points
 ]
grid_predictions_over = [
    predict(mach_over, Tables.table(reshape(collect(point), 1, 2)))[1] for
    point in grid_points
]
```

Make two contour plots using the grid predictions before and after oversampling


```julia
%%capture
p = contourf(petal_length_range, petal_width_range, grid_predictions,
    levels=3, color=:Set3_3, colorbar=false)
p_over = contourf(petal_length_range, petal_width_range, grid_predictions_over,
    levels=3, color=:Set3_3, colorbar=false)
```

Scatter plot the data before and after oversampling


```julia
labels = unique(y)
colors = Dict("setosa"=> "green", "versicolor" => "yellow",
              "virginica"=> "purple")

for label in labels
scatter!(p, X.petal_length[y. == label], X.petal_width[y. == label],
         color=colors[label], label=label,
         title="Before Oversampling")
scatter!(p_over, Xover.petal_length[yover. == label], Xover.petal_width[yover. == label],
         color=colors[label], label=label,
         title="After Oversampling")
end

plot_res = plot(p, p_over, layout=(1, 2), xlabel="petal length",
                ylabel="petal width", size=(900, 300))
savefig(plot_res, "./before-after-smote.png")

```

![](https://i.imgur.com/LMnKP9I.png)


Notice how the minority class was completely ignore prior to oversampling. Not all models and hyperparameter settings are this delicate to class imbalance.


## Effect of Ratios Hyperparameter

Now let's study the effect of the ratios hyperparameter. We will do this through an animated plot.


```julia
anim = @animate for versicolor_ratio ∈ 0.3:0.01:2
	# oversample
	Xover, yover =
		smote(X, y; k = 5, ratios = Dict("versicolor" => versicolor_ratio), rng = 42)

	# fit machine
	model = SVC(gamma = 0.01)
	mach_over = machine(model, Xover, yover)
	fit!(mach_over, verbosity = 0)

	# grid predictions
	grid_predictions_over = [
		predict(mach_over, Tables.table(reshape(collect(point), 1, 2)))[1] for
		point in grid_points
	]

	# plot
	p_over = contourf(petal_length_range, petal_width_range, grid_predictions_over,
		levels = 3, color = :Set3_3, colorbar = false)
	for label in labels
		scatter!(p_over, Xover.petal_length[yover.==label],
			Xover.petal_width[yover.==label],
			color = colors[label], label = label,
			title = "Oversampling versicolor with ratio $versicolor_ratio")
	end
	plot!(dpi = 150)
end
```


```julia
gif(anim, "./rose-animation.gif", fps=6)
println()
```

![abc](https://i.imgur.com/lxPhEke.gif)

Notice how setting ratios greedily can lead to overfitting.



