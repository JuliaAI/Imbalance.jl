
### SMOTEN with MLJ Interface

mutable struct SMOTEN{T,R<:Union{Integer,AbstractRNG}, I<:Integer} <: Static
    k::I
    ratios::T
    rng::R
    try_preserve_type::Bool
end;



"""
Check whether the given model hyperparameters are valid and clean them if necessary. 
"""
function MMI.clean!(s::SMOTEN)
    message = ""
    if s.k < 1
        throw((ERR_NONPOS_K(s.k)))
    end
    return message
end

"""
Initiate a SMOTEN model with the given hyper-parameters.
"""
function SMOTEN(;
    k::Integer = 5,
    ratios::Union{Nothing,AbstractFloat,Dict{T,<:AbstractFloat}} = 1.0,
    rng::Union{Integer,AbstractRNG} = default_rng(), try_preserve_type::Bool=true
) where {T}
    model = SMOTEN(k, ratios, rng, try_preserve_type)
    MMI.clean!(model)
    return model
end


"""
Oversample data X, y using SMOTEN
"""
function MMI.transform(s::SMOTEN, _, X, y)
    smoten(X, y; k = s.k, ratios = s.ratios, rng = s.rng, 
        try_preserve_type = s.try_preserve_type)
end
function MMI.transform(s::SMOTEN, _, X::AbstractMatrix{<:Integer}, y)
    smoten(X, y; k = s.k, ratios = s.ratios, rng = s.rng)
end


MMI.metadata_pkg(
    SMOTEN,
    name = "Imbalance",
    package_uuid = "c709b415-507b-45b7-9a3d-1767c89fde68",
    package_url = "https://github.com/JuliaAI/Imbalance.jl",
    is_pure_julia = true,
)


MMI.metadata_model(
    SMOTEN,
    input_scitype = Tuple{
                        Union{
                            Table(Finite),
                            AbstractMatrix{<:Finite}
                        }, 
                        AbstractVector
                    },
    output_scitype = Tuple{
        Union{
            Table(Finite),
            AbstractMatrix{<:Finite}
        }, 
        AbstractVector
    },
    load_path = "Imbalance.MLJ.SMOTEN"
)

function MMI.transform_scitype(s::SMOTEN)
    return Tuple{
        Union{Table(Finite),AbstractMatrix{<:Finite}},
        AbstractVector{<:Finite},
    }
end


"""
$(MMI.doc_header(SMOTEN))

`SMOTEN` implements the SMOTEN algorithm to correct for class imbalance as in
N. V. Chawla, K. W. Bowyer, L. O.Hall, W. P. Kegelmeyer,
“SMOTEN: synthetic minority over-sampling technique,”
Journal of artificial intelligence research, 321-357, 2002.


# Training data

In MLJ or MLJBase, wrap the model in a machine by

    mach = machine(model)

There is no need to provide any data here because the model is a static transformer.

Likewise, there is no need to `fit!(mach)`.

For default values of the hyper-parameters, model can be constructed by

    model = SMOTEN()


# Hyperparameters

- `k=5`: Number of nearest neighbors to consider in the SMOTEN algorithm.  Should be within
    the range `[1, n - 1]`, where `n` is the number of observations; otherwise set to the
    nearest of these two values.

$((COMMON_DOCS["RATIOS"]))

$((COMMON_DOCS["RNG"]))

# Transform Inputs

- `X`: A matrix of integers or a table with element [scitypes](https://juliaai.github.io/ScientificTypes.jl/) that subtype `Finite`. 
     That is, for table inputs each column should have either `OrderedFactor` or `Multiclass` as the element [scitype](https://juliaai.github.io/ScientificTypes.jl/).

- `y`: An abstract vector of labels (e.g., strings) that correspond to the observations in `X`


# Transform Outputs

$((COMMON_DOCS["OUTPUTS"]))

# Operations

- `transform(mach, X, y)`: resample the data `X` and `y` using SMOTEN, returning both the
  new and original observations


# Example

```julia
using MLJ
using ScientificTypes
import Imbalance

# set probability of each class
class_probs = [0.5, 0.2, 0.3]                         
num_rows = 100
num_continuous_feats = 0
# want two categorical features with three and two possible values respectively
num_vals_per_category = [3, 2]

# generate a table and categorical vector accordingly
X, y = Imbalance.generate_imbalanced_data(num_rows, num_continuous_feats; 
                                class_probs, num_vals_per_category, rng=42)                      
julia> Imbalance.checkbalance(y)
1: ▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇ 19 (39.6%) 
2: ▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇ 33 (68.8%) 
0: ▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇ 48 (100.0%) 

julia> ScientificTypes.schema(X).scitypes
(Count, Count)

# coerce to a finite scitype (multiclass or ordered factor)
X = coerce(X, autotype(X, :few_to_finite))

# load SMOTEN
SMOTEN = @load SMOTEN pkg=Imbalance

# wrap the model in a machine
oversampler = SMOTEN(k=5, ratios=Dict(0=>1.0, 1=> 0.9, 2=>0.8), rng=42)
mach = machine(oversampler)

# provide the data to transform (there is nothing to fit)
Xover, yover = transform(mach, X, y)

julia> Imbalance.checkbalance(yover)
2: ▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇ 38 (79.2%) 
1: ▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇ 43 (89.6%) 
0: ▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇ 48 (100.0%) 
```

"""
SMOTEN