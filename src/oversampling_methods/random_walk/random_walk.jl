
"""
Label encode and decode each column in a given table X
"""
random_walk_encoder(X) =
	generic_encoder(X; error_checker = check_scitypes_random_walk, return_cat_inds = true)
random_walk_decoder(X, d) = generic_decoder(X, d)

"""
Check that columns have the correct scientific types and if not, throw an error.

# Arguments
- `ncols`: Number of columns
- `cat_inds`: Indices of categorical columns
- `cont_inds`: Indices of continuous columns
- `types`: Type of each column
"""
function check_scitypes_random_walk(ncols, cat_inds, cont_inds, types)	
	bad_cols = setdiff(1:ncols, vcat(cat_inds, cont_inds))	# columns with wrong scitype
	if !isempty(bad_cols)
		throw(ArgumentError(ERR_BAD_MIXED_COL_TYPES(bad_cols, types[bad_cols])))
	end
end



"""
Choose a random point \$x\$ from the given observations matrix `X` and generate a new point \$z\$ that
in terms of the continuous part, satisfies \$z_{i} = x_{i} - \\frac{σ}{\\sqrt{N}} * N(0, 1)\$ and in terms
of the categorical part, satisfies \$z_{i} = c\$ where \$c\$ is a random category sampled from the approximate
distribution of the possible category values as estimated by `X[i, :]`.

# Arguments
- `X`: A matrix where each row is an observation
- `cont_inds`: A vector of indices of the continuous features
- `cat_inds`: A vector of indices of the categorical features
- `σ`: A vector of standard deviations for each continuous features
- `P`: A vector where each element is the approximate probability distribution vector of a categorical feature
- `rng`: Random number generator

# Returns
- `x_new`: A new observation generated by random walk oversampling
"""
function generate_new_random_walk_point(
	X::AbstractMatrix{<:AbstractFloat},
	cont_inds::AbstractVector{<:Int},
	cat_inds::AbstractVector{<:Int},
	σ::AbstractVector{<:AbstractFloat},
	P::AbstractVector{<:AbstractVector{<:AbstractFloat}};
	rng::AbstractRNG,
)
	# 1. Choose a random point from X (by index)
	ind = rand(rng, 1:size(X, 2))
	x_rand = X[:, ind]

	# 2. Find the continuous part of new point
	x_rand_cont = @view x_rand[cont_inds]
	n = size(X, 2)
	r = randn(rng, length(x_rand_cont))
	x_new_cont = @. x_rand_cont - r * σ/ sqrt(n)

	# 3. Find the categorical part of new point 
	x_new_cat = [
		sample(rng, collect(1:length(P[i])), ProbabilityWeights(P[i], 1.0)) for
		i in eachindex(1:length(cat_inds))
	]

	#4. Make the final vector
	x_new = fill(0.0, size(X, 1))
	x_new[cont_inds] = x_new_cont
	x_new[cat_inds] = x_new_cat

	return x_new
end



"""
Assuming that all the observations in the observation matrix X belong to the same class,
use random walk oversampling to generate `n` new observations for that class.

# Arguments
- `X`: A matrix where each row is an observation
- `n`: Number of new observations to generate
- `cont_inds`: A vector of indices of the continuous features
- `cat_inds`: A vector of indices of the categorical features
- `rng`: Random number generator

# Returns
- `Xnew`: A matrix where each row is a new observation generated by random walk oversampling
"""
function random_walk_per_class(
    X::AbstractMatrix{<:AbstractFloat},
    n::Integer,
    cont_inds::AbstractVector{<:Int},
    cat_inds::AbstractVector{<:Int};
    rng::AbstractRNG = default_rng(),
)
    # compute continuous feature statistics
    X_cont = X[cont_inds, :]
    σ = vec(std(X_cont, dims = 2, corrected=false))

    # compute categorical feature statistics
    X_cat = Int32.(X[cat_inds, :])
    P = [proportions(x) for x in eachrow(X_cat)]

    # Generate n new observations
    Xnew = zeros(Float32, size(X, 1), n)
    p = Progress(n)
    for i=1:n
        Xnew[:, i] = generate_new_random_walk_point(X, cont_inds, cat_inds, σ, P; rng)
        next!(p)
    end
    return Xnew
end


"""
	random_walk_oversample(
		X, y, cat_inds;
		ratios=1.0, rng=default_rng(),
		try_preserve_type=true
	)

# Description

Oversamples a dataset using random walk oversampling as presented in [1]. 

# Positional Arguments

- `X`: A matrix of floats or a table with element [scitypes](https://juliaai.github.io/ScientificTypes.jl/) that subtype `Union{Finite, Infinite}`. 
     Elements in nominal columns should subtype `Finite` (i.e., have scitype `OrderedFactor` or `Multiclass`) and
     elements in continuous columns should subtype `Infinite` (i.e., have scitype `Count` or `Continuous`).

- `y`: An abstract vector of labels (e.g., strings) that correspond to the observations in `X`

- `cat_inds::AbstractVector{<:Int}`: A vector of the indices of the nominal features. Supplied only if `X` is a matrix.
        Otherwise, they are inferred from the table's scitypes.


# Keyword Arguments

$(COMMON_DOCS["RATIOS"])

$(COMMON_DOCS["RNG"])

$(COMMON_DOCS["TRY_PRESERVE_TYPE"])

# Returns

$(COMMON_DOCS["OUTPUTS"])

# Example
```@repl
using Imbalance

# set probability of each class
class_probs = [0.5, 0.2, 0.3]                         
num_rows = 100
num_continuous_feats = 3
# want two categorical features with three and two possible values respectively
num_vals_per_category = [3, 2]

# generate a table and categorical vector accordingly
X, y = generate_imbalanced_data(num_rows, num_continuous_feats; 
                                class_probs, num_vals_per_category, rng=42)    
								                  
julia> Imbalance.checkbalance(y)
1: ▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇ 19 (39.6%) 
2: ▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇ 33 (68.8%) 
0: ▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇ 48 (100.0%) 

julia> ScientificTypes.schema(X).scitypes
(Continuous, Continuous, Continuous, Continuous, Continuous)
# coerce nominal columns to a finite scitype (multiclass or ordered factor)
X = coerce(X, :Column4=>Multiclass, :Column5=>Multiclass)

# apply random walk oversampling
Xover, yover = random_walk_oversample(X, y; 
                                      ratios = Dict(0=>1.0, 1=> 0.9, 2=>0.8), rng = 42)

julia> Imbalance.checkbalance(yover)
2: ▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇ 38 (79.2%) 
1: ▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇ 43 (89.6%) 
0: ▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇ 48 (100.0%) 
```
# MLJ Model Interface

Simply pass the keyword arguments while initiating the `RandomWalkOversampling` model and pass the 
	positional arguments (excluding `cat_inds`) to the `transform` method. 

```julia
using MLJ
RandomWalkOversampler = @load RandomWalkOversampler pkg=Imbalance

# Wrap the model in a machine
oversampler = RandomWalkOversampler(ratios=Dict(0=>1.0, 1=> 0.9, 2=>0.8), rng=42)
mach = machine(oversampler)

# Provide the data to transform (there is nothing to fit)
Xover, yover = transform(mach, X, y)
```
You can read more about this `MLJ` interface [here](). Note that only `Table` input is supported by the MLJ interface for this method.


# TableTransforms Interface

This interface assumes that the input is one table `Xy` and that `y` is one of the columns. Hence, an integer `y_ind`
	must be specified to the constructor to specify which column `y` is followed by other keyword arguments. 
	Only `Xy` is provided while applying the transform.

```julia
using Imbalance
using ScientificTypes
using Imbalance.TableTransforms

# Generate imbalanced data
num_rows = 100
num_continuous_feats = 3
y_ind = 2

# generate a table and categorical vector accordingly
Xy, _ = generate_imbalanced_data(num_rows, num_continuous_feats; insert_y=y_ind,
                                class_probs= [0.5, 0.2, 0.3], num_vals_per_category=[3, 2],
                                 rng=42)  

# Table must have only finite or continuous scitypes                                
Xy = coerce(Xy, :Column2=>Multiclass, :Column5=>Multiclass, :Column6=>Multiclass)

# Initiate Random Walk Oversampler model
oversampler = RandomWalkOversampler(y_ind;
                                    ratios=Dict(1=>1.0, 2=> 0.9, 3=>0.9), rng=42)
Xyover = Xy |> oversampler                               
# equivalently if TableTransforms is used
Xyover, cache = TableTransforms.apply(oversampler, Xy)    # equivalently
```
# Illustration
A full basic example along with an animation can be found [here](https://githubtocolab.com/JuliaAI/Imbalance.jl/blob/dev/examples/oversample_randomwalk.ipynb). 
    You may find more practical examples in the [walkthrough](https://juliaai.github.io/Imbalance.jl/dev/examples/) 
    section which also explains running code on Google Colab.

# References
[1] Zhang, H., & Li, M. (2014). RWO-Sampling: A random walk over-sampling approach to imbalanced data classification. 
Information Fusion, 25, 4-20.
"""
function random_walk_oversample(
    X::AbstractMatrix{<:AbstractFloat},
    y::AbstractVector,
    cat_inds::AbstractVector{<:Int};
    ratios = 1.0,
    rng::Union{AbstractRNG,Integer} = default_rng(),
	try_preserve_type::Bool = true,
)
    rng = rng_handler(rng)
    # implictly infer the continuous indices
    cont_inds = setdiff(1:size(X, 2), cat_inds)
    Xover, yover =
        generic_oversample(X, y, random_walk_per_class, cont_inds, cat_inds; ratios, rng)
    return Xover, yover
end

# dispatch for when X is a table
function random_walk_oversample(
    X,
    y::AbstractVector;
    ratios = 1.0,
    rng::Union{AbstractRNG,Integer} = default_rng(),
	try_preserve_type::Bool = true,
)
	Xover, yover = tablify(
		random_walk_oversample,
		X,
		y;
		try_preserve_type = try_preserve_type,
		encode_func = random_walk_encoder,
		decode_func = random_walk_decoder,
		ratios,
		rng,
	)
	return Xover, yover
end

# dispatch for when X is a table and y is one of its columns
function random_walk_oversample(
	Xy,
	y_ind::Integer;
	ratios = 1.0,
	rng::Union{AbstractRNG, Integer} = default_rng(),
	try_preserve_type::Bool = true,
)
	Xyover = tablify(
		random_walk_oversample,
		Xy,
		y_ind;
		try_preserve_type = try_preserve_type,
		encode_func = random_walk_encoder,
		decode_func = random_walk_decoder,
		ratios,
		rng,
	)
	return Xyover
end