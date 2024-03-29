### SMOTENC with MLJ Interface
mutable struct SMOTENC{T,R<:Union{Integer,AbstractRNG}, S<:AbstractString, I<:Integer} <: Static
    k::I
    ratios::T
    knn_tree::S
    rng::R
    try_preserve_type::Bool
end


"""
Check whether the given model hyperparameters are valid and clean them if necessary. 
"""
function MMI.clean!(s::SMOTENC)
    message = ""
    if s.k < 1
        throw((ERR_NONPOS_K(s.k)))
    end
    (s.knn_tree ∈ ["Ball", "Brute"]) || throw((ERR_WRNG_TREE(knn_tree)))
    return message
end


"""
Initiate a SMOTENC model with the given hyper-parameters.
"""
function SMOTENC(;
    k::Integer = 5,
    ratios::Union{Nothing,AbstractFloat,Dict{T,<:AbstractFloat}} =1.0,
    knn_tree::AbstractString = "Brute",
    rng::Union{Integer,AbstractRNG} = default_rng(),
    try_preserve_type::Bool=true
) where {T}
    model = SMOTENC(k, ratios, knn_tree, rng, try_preserve_type)
    MMI.clean!(model)
    return model
end




"""
Oversample data X, y using SMOTENC
"""
function MMI.transform(s::SMOTENC, _, X, y)
    smotenc(X, y; k = s.k, ratios = s.ratios, knn_tree=s.knn_tree, rng = s.rng, try_preserve_type=s.try_preserve_type)
end




MMI.metadata_pkg(
    SMOTENC,
    name = "Imbalance",
    package_uuid = "c709b415-507b-45b7-9a3d-1767c89fde68",
    package_url = "https://github.com/JuliaAI/Imbalance.jl",
    is_pure_julia = true,
)

MMI.metadata_model(
    SMOTENC,
    input_scitype = Tuple{
        Table(Union{Infinite, Finite}),
        AbstractVector
    },
    output_scitype = Tuple{
        Table(Union{Infinite, Finite}),
        AbstractVector
    },
    load_path = "Imbalance.MLJ.SMOTENC"
)


function MMI.transform_scitype(s::SMOTENC)
    return Tuple{
        Union{
            Table(Union{Infinite,OrderedFactor,Multiclass}),
        },
        AbstractVector{<:Finite},
    }
end



"""
$(MMI.doc_header(SMOTENC))

`SMOTENC` implements the SMOTENC algorithm to correct for class imbalance as in
N. V. Chawla, K. W. Bowyer, L. O.Hall, W. P. Kegelmeyer,
“SMOTE: synthetic minority over-sampling technique,”
Journal of artificial intelligence research, 321-357, 2002.


# Training data

In MLJ or MLJBase, wrap the model in a machine by

    mach = machine(model)

There is no need to provide any data here because the model is a static transformer.

Likewise, there is no need to `fit!(mach)`.

For default values of the hyper-parameters, model can be constructed by

    model = SMOTENC()


# Hyperparameters

- `k=5`: Number of nearest neighbors to consider in the SMOTENC algorithm.  Should be within
    the range `[1, n - 1]`, where `n` is the number of observations; otherwise set to the
    nearest of these two values.

$((COMMON_DOCS["RATIOS"]))

- `knn_tree`: Decides the tree used in KNN computations. Either `"Brute"` or `"Ball"`.
    BallTree can be much faster but may lead to inaccurate results.

$((COMMON_DOCS["RNG"]))

# Transform Inputs

- `X`: A table with element [scitypes](https://juliaai.github.io/ScientificTypes.jl/) that subtype `Union{Finite, Infinite}`. 
     Elements in nominal columns should subtype `Finite` (i.e., have [scitype](https://juliaai.github.io/ScientificTypes.jl/) `OrderedFactor` or `Multiclass`) and
     elements in continuous columns should subtype `Infinite` (i.e., have [scitype](https://juliaai.github.io/ScientificTypes.jl/) `Count` or `Continuous`).

- `y`: An abstract vector of labels (e.g., strings) that correspond to the observations in `X`

# Transform Outputs

$((COMMON_DOCS["OUTPUTS"]))

# Operations

- `transform(mach, X, y)`: resample the data `X` and `y` using SMOTENC, returning both the
  new and original observations


# Example

```julia
using MLJ
using ScientificTypes
import Imbalance

# set probability of each class
class_probs = [0.5, 0.2, 0.3]                         
num_rows = 100
num_continuous_feats = 3
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
(Continuous, Continuous, Continuous, Continuous, Continuous)
# coerce nominal columns to a finite scitype (multiclass or ordered factor)
X = coerce(X, :Column4=>Multiclass, :Column5=>Multiclass)

# load SMOTE-NC
SMOTENC = @load SMOTENC pkg=Imbalance

# wrap the model in a machine
oversampler = SMOTENC(k=5, ratios=Dict(0=>1.0, 1=> 0.9, 2=>0.8), rng=42)
mach = machine(oversampler)

# provide the data to transform (there is nothing to fit)
Xover, yover = transform(mach, X, y)

julia> Imbalance.checkbalance(yover)
2: ▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇ 38 (79.2%) 
1: ▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇ 43 (89.6%) 
0: ▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇ 48 (100.0%) 
```
"""
SMOTENC

