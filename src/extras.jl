
"""
    generate_imbalanced_data(
        num_rows, num_continuous_feats;
        means=nothing, min_sep=1.0, stds=nothing,
        num_vals_per_category = [],
        class_probs = [0.8, 0.2],
        type= "ColTable", insert_y= nothing,
        rng= default_rng(),
    )

Generate `num_rows` observations with target `y` respecting given probabilities of 
    each class. Supports generating continuous features with a specific mean and variance
    and categorical features given the number of levels in each variable.

# Arguments
- `num_rows::Integer`: Number of observations to generate
- `num_continuous_feats::Integer`: Number of continuous features to generate
- `means::AbstractVector=nothing`: A vector of means for each continuous feature (must be as long as `num_continuous_feats`). If `nothing`, then will
    be set randomly
- `min_sep::AbstractFloat=1.0`: Minimum distance between any two randomly chosen means. Will have no effect if the
    means are given.
- `stds::AbstractVector=nothing`: A vector of standard deviations for each continuous feature (must be as long as `num_continuous_feats`). If `nothing`, then
    will be set randomly
- `num_vals_per_category::AbstractVector=[]`: A vector of the number of levels of each extra categorical feature.
    the number of categorical features is inferred from this.
- `class_probs::AbstractVector{<:AbstractFloat}=[0.8, 0.2]`: A vector of probabilities of each class. The number of classes is inferred
    from this vector.
- `type::AbstractString="ColTable"`: Can be `"Matrix"` or `"ColTable"`. In the latter case, a named-tuple of vectors is returned.
- `insert_y::Integer=nothing`: If not nothing, insert the class labels column at the given index in the table
- `rng::Union{AbstractRNG, Integer}=default_rng()`: Random number generator. If integer then used as `seed` in `Random.Xoshiro(seed)` 

# Returns
- `X:`: A column table or matrix with generated imbalanced data with `num_rows` rows and 
    `num_continuous_feats + length(num_vals_per_category)` columns. If `insert_y` is specified as in integer
    then `y` is also inserted at the specified index as an extra column.
- `y::CategoricalArray`: An abstract vector of class labels with labels \$0\$, \$1\$, \$2\$, ..., \$k-1\$
    where `k=length(class_probs)`

# Example
```julia
using Imbalance
using Plots

num_rows = 500
num_features = 2
# generating continuous features given mean and std
X, y = generate_imbalanced_data(
	num_rows,
	num_features;
	means = [1.0, 4.0, [7.0 9.0]],
	stds = [1.0, [0.5 0.8], 2.0],
	class_probs=[0.5, 0.2, 0.3],
	type="Matrix",
	rng = 42,
)

p = plot()
[scatter!(p, X[:, 1][y.==yi], X[:, 2][y.==yi], label = "\$y=yi\$") for yi in unique(y)]

julia> plot(p)
```

![generated data](../../assets/gen_one.png)

```julia
# generating continuous features with random mean and std
X, y = generate_imbalanced_data(
	num_rows,
	num_features;
    min_sep=0.3,      
	class_probs=[0.5, 0.2, 0.3],
	type="Matrix",
	rng = 33,
)

p = plot()
[scatter!(p, X[:, 1][y.==yi], X[:, 2][y.==yi], label = "\$y=yi\$") for yi in unique(y)]

julia> plot(p)
```

![generated data](../../assets/gen_two.png)

```julia
num_rows = 500
num_features = 2
X, y = generate_imbalanced_data(
	num_rows,
	num_features;
    num_vals_per_category = [3, 5, 2],
	class_probs=[0.9, 0.1],
	insert_y=4,
	type="ColTable",
	rng = 33,
)

julia> X
(Column1 = [0.883, 0.9, 0.577  …  0.887,],
 Column2 = [0.578, 0.718, 0.378  …  0.573,],
 Column3 = [2.0, 2.0, 3.0, …  2.0,],
 Column4 = [0.0, 0.0, 0.0, …  0.0,],
 Column5 = [2.0, 3.0, 4.0, …  4.0,],
 Column6 = [1.0, 1.0, 2.0, …  1.0,],)
```
"""
function generate_imbalanced_data(
    num_rows::Integer,
    num_continuous_feats::Integer;
    means=nothing,
    min_sep::AbstractFloat=1.0,
    stds=nothing,
    num_vals_per_category::AbstractVector = [],
    class_probs::AbstractVector{<:AbstractFloat} = [0.8, 0.2],
    type::AbstractString = "ColTable",
    insert_y= nothing,
    rng::Union{AbstractRNG, Integer} = default_rng(),
)
    rng = rng_handler(rng)
    
    # Generate y as a categorical array with classes 0, 1, 2, ..., k-1
    cum_class_probs = cumsum(class_probs)
    rands = rand(rng, num_rows)
    y = CategoricalArray([findfirst(x -> rands[i] <= x, cum_class_probs) - 1 for i in 1:num_rows])

    # if no continuous features, go for an integer matrix
    if num_continuous_feats > 0
        Xc = Matrix{Float64}(undef, num_rows, num_continuous_feats)
        # set the previous means to -∞ so min_sep is satisfied for first random selection
        μ_prevs =  [[-Inf for i in 1:num_continuous_feats]']
        # for each class generate data following means[i], stds[i] or randomly choose them
        for i in eachindex(class_probs)
            t = 1                   # will help min_sep be sastisfied
            if isnothing(means) 
                # choose a random μ to satisfy min_sep
                μ = rand(rng, num_continuous_feats)' 
                while  any([sqrt(sum((μ-μ_prev).^2)) < min_sep for μ_prev in μ_prevs])
                    μ = rand(rng, num_continuous_feats)' * ℯ^0.5t
                    t+=1
                end
                push!(μ_prevs, μ) 
            else
                μ = means[i]
            end
            σ = isnothing(stds) ? rand(rng, num_continuous_feats)' * 0.3 :
            stds[i]
            # put the generated data at the corresponding indices to y
            class_inds = BitVector(y .== (i-1))
            Xc[class_inds, :] = round.((randn(rng, sum(class_inds), num_continuous_feats) .* σ) .+ μ, digits=3)
        end
    else
        Xc = Matrix{Int64}(undef, num_rows, 0)
    end

    # generate categorical data as integer
    for num_levels in num_vals_per_category
        Xc = hcat(Xc, rand(rng, 1:num_levels, num_rows))
    end

    # insert if needed
    if !isnothing(insert_y)
        Xc = hcat(Xc[:, 1:insert_y-1], y, Xc[:, insert_y:end])
    end

    # support for different types; not exposed but used for testing.
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

A visual version of `StatsBase.countmap` that returns nothing and prints how 
    many observations in the dataset belong to each class and their percentage
    relative to the size of majority or minority class.

# Arguments
- `y::AbstractVector`: A vector of categorical values to test for imbalance
- `reference="majority"`: Either `"majority"` or `"minority"` and decides whether the percentage should be
    relative to the size of majority or minority class.

# Example
```julia
num_rows = 50000
num_features = 2
X, y = generate_imbalanced_data(
	num_rows,
	num_features;
	class_probs=[0.8, 0.2],
	type="Matrix",
	rng = 42,
)

julia> Imbalance.checkbalance(y; ref="majority")
1: ▇▇▇▇▇▇▇▇▇▇▇▇▇ 10034 (25.1%) 
0: ▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇ 39966 (100.0%) 

julia> Imbalance.checkbalance(y; ref="minority")
1: ▇▇▇▇▇▇▇▇▇▇▇▇▇ 10034 (100.0%) 
0: ▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇ 39966 (398.3%) 
```
"""
function checkbalance(y; ref = "majority")
    counts = countmap(y)
    sorted_counts = sort(collect(counts), by = x -> x[2])
    (ref in ["majority", "minority"]) || error("Invalid reference")
    ref_class_count =
        (ref == "majority") ? maximum(values(counts)) : minimum(values(counts))
    majority_class_count = maximum(values(counts))      # in call cases, longer bar for max
    longest_label_length = maximum(length.(string.(keys(counts))))  

    for (key, count) in sorted_counts
        percentage = round(100 * count / ref_class_count, digits = 1)
        bar_length = round(Int, count * 50 / majority_class_count)
        bar = "▇"^bar_length
        padding = " "^(longest_label_length - length(string.(key)))
        println("$key:$padding $bar $count ($percentage%) ")
    end
end