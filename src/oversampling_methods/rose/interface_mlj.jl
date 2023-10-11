### ROSE MLJ Interface
# interface struct
mutable struct ROSE{T,R<:Union{Integer,AbstractRNG}, F<:AbstractFloat} <: MMI.Static
    s::F
    ratios::T
    rng::R
    try_preserve_type::Bool
end;



"""
Check whether the given model hyperparameters are valid and clean them if necessary. 
"""
function MMI.clean!(r::ROSE)
    message = ""
    if r.s < 0
        throw((ERR_NONNEG_S(r.s)))
    end
    return message
end

"""
Initiate a ROSE model with the given hyper-parameters.
"""
function ROSE(;
    s::AbstractFloat = 1.0,
    ratios::Union{Nothing,AbstractFloat,Dict{T,<:AbstractFloat}} = 1.0,
    rng::Union{Integer,AbstractRNG} = default_rng(), try_preserve_type::Bool = true,
) where {T}
    model = ROSE(s, ratios, rng, try_preserve_type)
    MMI.clean!(model)
    return model
end

"""
Oversample data X, y using ROSE
"""
function MMI.transform(r::ROSE, _, X, y)
    rose(X, y; s = r.s, ratios = r.ratios, rng = r.rng, 
        try_preserve_type = r.try_preserve_type)
end
function MMI.transform(r::ROSE, _, X::AbstractMatrix{<:Real}, y)
    rose(X, y; s = r.s, ratios = r.ratios, rng = r.rng,)
end

MMI.metadata_pkg(
    ROSE,
    name = "Imbalance",
    package_uuid = "c709b415-507b-45b7-9a3d-1767c89fde68",
    package_url = "https://github.com/JuliaAI/Imbalance.jl",
    is_pure_julia = true,
)

MMI.metadata_model(
    ROSE,
    input_scitype = Union{Table(Continuous),AbstractMatrix{Continuous}},
    output_scitype = Union{Table(Continuous),AbstractMatrix{Continuous}},
    target_scitype = AbstractVector,
    load_path = "Imbalance.MLJ.ROSE"
)
function MMI.transform_scitype(s::ROSE)
    return Tuple{
        Union{Table(Continuous),AbstractMatrix{Continuous}},
        AbstractVector{<:Finite},
    }
end


"""
$(MMI.doc_header(ROSE))

`ROSE` implements the ROSE (Random Oversampling Examples) algorithm to 
    correct for class imbalance as in G Menardi, N. Torelli, “Training and assessing 
    classification rules with imbalanced data,” 
    Data Mining and Knowledge Discovery, 28(1), pp.92-122, 2014.


# Training data

In MLJ or MLJBase, wrap the model in a machine by
    mach = machine(model)

There is no need to provide any data here because the model is a static transformer.

Likewise, there is no need to `fit!(mach)`. 

For default values of the hyper-parameters, model can be constructed by
    model = ROSE()


# Hyperparameters

- `s::float`: A parameter that proportionally controls the bandwidth of the Gaussian kernel

$((COMMON_DOCS["RATIOS"]))

$((COMMON_DOCS["RNG"]))

# Transform Inputs

$((COMMON_DOCS["INPUTS"]))

# Transform Outputs

$((COMMON_DOCS["OUTPUTS"]))

# Operations

- `transform(mach, X, y)`: resample the data `X` and `y` using ROSE, returning both the
  new and original observations

# Example

```julia
using MLJ
import Imbalance

# set probability of each class
class_probs = [0.5, 0.2, 0.3]                         
num_rows, num_continuous_feats = 100, 5
# generate a table and categorical vector accordingly
X, y = Imbalance.generate_imbalanced_data(num_rows, num_continuous_feats; 
                                class_probs, rng=42)  

julia> Imbalance.checkbalance(y)
1: ▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇ 19 (39.6%) 
2: ▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇ 33 (68.8%) 
0: ▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇ 48 (100.0%) 

# load ROSE
ROSE = @load ROSE pkg=Imbalance

# wrap the model in a machine
oversampler = ROSE(s=0.3, ratios=Dict(0=>1.0, 1=> 0.9, 2=>0.8), rng=42)
mach = machine(oversampler)

# provide the data to transform (there is nothing to fit)
Xover, yover = transform(mach, X, y)

julia> Imbalance.checkbalance(yover)
2: ▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇ 38 (79.2%) 
1: ▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇ 43 (89.6%) 
0: ▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇ 48 (100.0%) 
```

"""
ROSE
