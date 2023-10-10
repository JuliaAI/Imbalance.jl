
### ClusterUndersampler with MLJ Interface
# interface struct
mutable struct ClusterUndersampler{
	S <: AbstractString,
	T,
	I <: Integer,
	R <: Union{AbstractRNG, Integer},
} <: Static
	mode::S
	ratios::T
	maxiter::I
	rng::R
	try_preserve_type::Bool
end;

"""
Initiate a cluster undersampling model with the given hyper-parameters.
"""
function ClusterUndersampler(;
	mode::AbstractString = "nearest",
	ratios::Union{Nothing, AbstractFloat, Dict{T, <:AbstractFloat}} = 1.0,
	maxiter::Integer = 100,
    rng::Union{Integer, AbstractRNG} = default_rng(),
	try_preserve_type::Bool = true,
) where {T}
	model = ClusterUndersampler(mode, ratios, maxiter, rng, try_preserve_type)
	return model
end

"""
Undersample data X, y 
"""
function MMI.transform(r::ClusterUndersampler, _, X, y)
	return cluster_undersample(
		X,
		y;
		mode = r.mode,
		ratios = r.ratios,
		maxiter = r.maxiter,
		rng = r.rng,
		try_preserve_type = r.try_preserve_type,
	)
end
function MMI.transform(r::ClusterUndersampler, _, X::AbstractMatrix{<:Real}, y)
	return cluster_undersample(
		X,
		y;
		mode = r.mode,
		ratios = r.ratios,
		maxiter = r.maxiter,
		rng = r.rng,
	)
end

MMI.metadata_pkg(
	ClusterUndersampler,
	name = "Imbalance",
	package_uuid = "c709b415-507b-45b7-9a3d-1767c89fde68",
	package_url = "https://github.com/JuliaAI/Imbalance.jl",
	is_pure_julia = true,
)

MMI.metadata_model(
	ClusterUndersampler,
	input_scitype = Union{Table(Continuous), AbstractMatrix{Continuous}},
	output_scitype = Union{Table(Continuous), AbstractMatrix{Continuous}},
	target_scitype = AbstractVector,
	load_path = "Imbalance.MLJ.ClusterUndersampler" 
)

function MMI.transform_scitype(s::ClusterUndersampler)
	return Tuple{
		Union{Table(Continuous), AbstractMatrix{Continuous}},
		AbstractVector{<:Finite},
	}
end

"""
$(MMI.doc_header(ClusterUndersampler))

`ClusterUndersampler` implements clustering undersampling as presented in Wei-Chao, L., Chih-Fong, T., Ya-Han, H., & Jing-Shang, J. (2017). 
  Clustering-based undersampling in class-imbalanced data. Information Sciences, 409–410, 17–26. with K-means as
  the clustering algorithm.


# Training data

In MLJ or MLJBase, wrap the model in a machine by
	mach = machine(model)

There is no need to provide any data here because the model is a static transformer.

Likewise, there is no need to `fit!(mach)`. 

For default values of the hyper-parameters, model can be constructed with `model = ClusterUndersampler()`.
	

# Hyperparameters

- `mode::AbstractString="nearest`: If `"center"` then the undersampled data will consist of the centriods of 
	each cluster found; if `"nearest"` then it will consist of the nearest neighbor of each centroid.

$(COMMON_DOCS["RATIOS-UNDERSAMPLE"])

- `maxiter::Integer=100`: Maximum number of iterations to run K-means

- `rng::Integer=42`: Random number generator seed. Must be an integer.

# Transform Inputs

$((COMMON_DOCS["INPUTS"]))

# Transform Outputs

$(COMMON_DOCS["OUTPUTS-UNDER"])

# Operations

- `transform(mach, X, y)`: resample the data `X` and `y` using ClusterUndersampler, returning the undersampled
  versions


# Example

```
using MLJ
import Imbalance

# set probability of each class
class_probs = [0.5, 0.2, 0.3]                         
num_rows, num_continuous_feats = 100, 5
# generate a table and categorical vector accordingly
X, y = Imbalance.generate_imbalanced_data(num_rows, num_continuous_feats; 
                                class_probs, rng=42)   
                                                    
julia> Imbalance.checkbalance(y; ref="minority")
 1: ▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇ 19 (100.0%) 
 2: ▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇ 33 (173.7%) 
 0: ▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇ 48 (252.6%) 

# load cluster_undersampling
ClusterUndersampler = @load ClusterUndersampler pkg=Imbalance

# wrap the model in a machine
undersampler = ClusterUndersampler(mode="nearest", 
                                   ratios=Dict(0=>1.0, 1=> 1.0, 2=>1.0), rng=42)
mach = machine(undersampler)

# provide the data to transform (there is nothing to fit)
X_under, y_under = transform(mach, X, y)

                                       
julia> Imbalance.checkbalance(y_under; ref="minority")
0: ▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇ 19 (100.0%) 
2: ▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇ 19 (100.0%) 
1: ▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇ 19 (100.0%)
```

"""
ClusterUndersampler
