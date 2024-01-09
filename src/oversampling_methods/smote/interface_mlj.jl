
### SMOTE with MLJ Interface

mutable struct SMOTE{T,R<:Union{Integer,AbstractRNG}, I<:Integer} <: Static
    k::I
    ratios::T
    rng::R
    try_preserve_type::Bool
end;



"""
Check whether the given model hyperparameters are valid and clean them if necessary. 
"""
function MMI.clean!(s::SMOTE)
    message = ""
    if s.k < 1
        throw((ERR_NONPOS_K(s.k)))
    end
    return message
end




"""
Initiate a SMOTE model with the given hyper-parameters.
"""
function SMOTE(;
    k::Integer = 5,
    ratios::Union{Nothing,AbstractFloat,Dict{T,<:AbstractFloat}} = 1.0,
    rng::Union{Integer,AbstractRNG} = default_rng(), try_preserve_type::Bool=true
) where {T}
    model = SMOTE(k, ratios, rng, try_preserve_type)
    MMI.clean!(model)
    return model
end

"""
Oversample data X, y using SMOTE
"""
function MMI.transform(s::SMOTE, _, X, y)
    smote(X, y; k = s.k, ratios = s.ratios, rng = s.rng, 
        try_preserve_type = s.try_preserve_type)
end
function MMI.transform(s::SMOTE, _, X::AbstractMatrix{<:Real}, y)
    smote(X, y; k = s.k, ratios = s.ratios, rng = s.rng)
end


MMI.metadata_pkg(
    SMOTE,
    name = "Imbalance",
    package_uuid = "c709b415-507b-45b7-9a3d-1767c89fde68",
    package_url = "https://github.com/JuliaAI/Imbalance.jl",
    is_pure_julia = true,
)

MMI.metadata_model(
    SMOTE,
    input_scitype = Tuple{
                        Union{
                            Table(Continuous),
                            AbstractMatrix{Continuous}
                        }, 
                        AbstractVector
                    },
    output_scitype = Tuple{
        Union{
            Table(Continuous),
            AbstractMatrix{Continuous}
        }, 
        AbstractVector
    },
    load_path = "Imbalance.MLJ.SMOTE"
)
function MMI.transform_scitype(s::SMOTE)
    return Tuple{
        Union{Table(Continuous),AbstractMatrix{Continuous}},
        AbstractVector{<:Finite},
    }
end



"""
$(MMI.doc_header(SMOTE))

`SMOTE` implements the SMOTE algorithm to correct for class imbalance as in
N. V. Chawla, K. W. Bowyer, L. O.Hall, W. P. Kegelmeyer,
“SMOTE: synthetic minority over-sampling technique,”
Journal of artificial intelligence research, 321-357, 2002.


# Training data

In MLJ or MLJBase, wrap the model in a machine by

    mach = machine(model)

There is no need to provide any data here because the model is a static transformer.

Likewise, there is no need to `fit!(mach)`.

For default values of the hyper-parameters, model can be constructed by

    model = SMOTE()


# Hyperparameters

- `k=5`: Number of nearest neighbors to consider in the SMOTE algorithm.  Should be within
    the range `[1, n - 1]`, where `n` is the number of observations; otherwise set to the
    nearest of these two values.

$((COMMON_DOCS["RATIOS"]))

$((COMMON_DOCS["RNG"]))

# Transform Inputs

$((COMMON_DOCS["INPUTS"]))

# Transform Outputs

$((COMMON_DOCS["OUTPUTS"]))

# Operations

- `transform(mach, X, y)`: resample the data `X` and `y` using SMOTE, returning both the
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

# load SMOTE
SMOTE = @load SMOTE pkg=Imbalance

# wrap the model in a machine
oversampler = SMOTE(k=5, ratios=Dict(0=>1.0, 1=> 0.9, 2=>0.8), rng=42)
mach = machine(oversampler)

# provide the data to transform (there is nothing to fit)
Xover, yover = transform(mach, X, y)

julia> Imbalance.checkbalance(yover)
2: ▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇ 38 (79.2%) 
1: ▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇ 43 (89.6%) 
0: ▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇ 48 (100.0%) 

```

"""
SMOTE
