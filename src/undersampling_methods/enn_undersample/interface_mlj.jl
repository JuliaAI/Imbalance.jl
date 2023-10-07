
### ENNUndersampler with MLJ Interface
# interface struct
mutable struct ENNUndersampler{
	T,
	S <: AbstractString,
	I <: Integer,
	R <: Union{Integer, AbstractRNG},
} <: Static
	k::I
	keep_condition::S
	min_ratios::T
	force_min_ratios::Bool
	rng::R
	try_preserve_type::Bool
end;

"""
Initiate a ENN undersampling model with the given hyper-parameters.
"""
function ENNUndersampler(;
	k::Integer = 5,
	keep_condition::AbstractString = "mode",
	min_ratios::Union{Nothing, AbstractFloat, Dict{T, <:AbstractFloat}} = 1.0,
	force_min_ratios::Bool = false,
    rng::Union{AbstractRNG, Integer} = default_rng(),
	try_preserve_type::Bool = true,
) where {T}
	model = ENNUndersampler(
		k,
		keep_condition,
		min_ratios,
		force_min_ratios,
		rng,
		try_preserve_type,
	)
	return model
end

"""
Undersample data X, y 
"""
function MMI.transform(r::ENNUndersampler, _, X, y)
	return enn_undersample(
		X,
		y;
		k = r.k,
		keep_condition = r.keep_condition,
		min_ratios = r.min_ratios,
		force_min_ratios = r.force_min_ratios,
		rng = r.rng,
		try_preserve_type = r.try_preserve_type,
	)
end
function MMI.transform(r::ENNUndersampler, _, X::AbstractMatrix{<:Real}, y)
	return enn_undersample(
		X,
		y;
		k = r.k,
		keep_condition = r.keep_condition,
		min_ratios = r.min_ratios,
		force_min_ratios = r.force_min_ratios,
		rng = r.rng,
	)
end

MMI.metadata_pkg(
	ENNUndersampler,
	name = "Imbalance",
	package_uuid = "c709b415-507b-45b7-9a3d-1767c89fde68",
	package_url = "https://github.com/JuliaAI/Imbalance.jl",
	is_pure_julia = true,
)

MMI.metadata_model(
	ENNUndersampler,
	input_scitype = Union{Table(Continuous)},
	output_scitype = Union{Table(Continuous)},
	target_scitype = AbstractVector,
	load_path = "Imbalance.MLJ.ENNUndersampler" 
)
function MMI.transform_scitype(s::ENNUndersampler)
	return Tuple{
		Union{Table(Continuous), AbstractMatrix{Continuous}},
		AbstractVector{<:Finite},
	}
end

"""
$(MMI.doc_header(ENNUndersampler))

`ENNUndersampler` undersamples a dataset by removing ("cleaning") points that violate a certain condition such as
  having a different class compared to the majority of the neighbors as proposed in Dennis L Wilson. 
  Asymptotic properties of nearest neighbor rules using edited data. IEEE Transactions on Systems, Man, 
  and Cybernetics, pages 408–421, 1972.


# Training data

In MLJ or MLJBase, wrap the model in a machine by
	mach = machine(model)

There is no need to provide any data here because the model is a static transformer.

Likewise, there is no need to `fit!(mach)`. 

For default values of the hyper-parameters, model can be constructed by
	model = ENNUndersampler()
	

# Hyperparameters

$(COMMON_DOCS["K"])

- `keep_condition::AbstractString="mode"`: The condition that leads to cleaning a point upon violation. Takes one of `"exists"`, `"mode"`, `"only mode"` and `"all"`
	- `"exists"`: the point has at least one neighbor from the same class
	- `"mode"`: the class of the point is one of the most frequent classes of the neighbors (there may be many)
	- `"only mode"`: the class of the point is the single most frequent class of the neighbors
	- `"all"`: the class of the point is the same as all the neighbors

$(COMMON_DOCS["MIN-RATIOS-UNDERSAMPLE"])

$(COMMON_DOCS["FORCE-MIN-RATIOS"])

$(COMMON_DOCS["RNG"])

$(COMMON_DOCS["TRY_PRESERVE_TYPE"])

# Transform Inputs

$((COMMON_DOCS["INPUTS"]))

# Transform Outputs

$(COMMON_DOCS["OUTPUTS-UNDER"])

# Operations

- `transform(mach, X, y)`: resample the data `X` and `y` using ENNUndersampler, returning the undersampled
  versions


# Example

```
using MLJ
import Imbalance

# set probability of each class
class_probs = [0.5, 0.2, 0.3]                         
num_rows, num_continuous_feats = 100, 5
# generate a table and categorical vector accordingly
X, y = generate_imbalanced_data(num_rows, num_continuous_feats; 
                                min_sep=0.01, stds=[3.0 3.0 3.0], class_probs, rng=42)     

julia> checkbalance(y; ref="minority")
1: ▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇ 19 (100.0%) 
2: ▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇ 33 (173.7%) 
0: ▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇ 48 (252.6%) 

# load ENN model type:
ENNUndersampler = @load ENNUndersampler pkg=Imbalance

# Underample the majority classes to  sizes relative to the minority class:
undersampler = ENNUndersampler(min_ratios=0.5, rng=42)
mach = machine(undersampler)
X_under, y_under = transform(mach, X, y)

julia> checkbalance(y_under; ref="minority")
2: ▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇ 10 (100.0%) 
1: ▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇ 10 (100.0%) 
0: ▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇ 24 (240.0%) 
```

"""
ENNUndersampler

