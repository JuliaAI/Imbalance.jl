
### BorderlineSMOTE1 with MLJ Interface

mutable struct BorderlineSMOTE1{T,R<:Union{Integer,AbstractRNG}, I<:Integer} <: Static
    m::I
    k::I
    ratios::T
    rng::R
    try_preserve_type::Bool
    verbosity::I
end;



"""
Check whether the given model hyperparameters are valid and clean them if necessary. 
"""
function MMI.clean!(s::BorderlineSMOTE1)
    message = ""
    if s.k < 1
        throw((ERR_NONPOS_K(s.k)))
    end
    if s.m < 1
        throw((ERR_NONPOS_K(s.m)))
    end
    return message
end




"""
Initiate a BorderlineSMOTE1 model with the given hyper-parameters.
"""
function BorderlineSMOTE1(;
    m::Integer = 5,
    k::Integer = 5,
    ratios::Union{Nothing,AbstractFloat,Dict{T,<:AbstractFloat}} = 1.0,
    rng::Union{Integer,AbstractRNG} = default_rng(), try_preserve_type::Bool=true, verbosity::Integer=1
) where {T}
    model = BorderlineSMOTE1(m, k, ratios, rng, try_preserve_type, verbosity)
    MMI.clean!(model)
    return model
end

"""
Oversample data X, y using BorderlineSMOTE1
"""
function MMI.transform(s::BorderlineSMOTE1, _, X, y)
    borderline_smote1(X, y; m = s.m, k = s.k, ratios = s.ratios, rng = s.rng, 
        try_preserve_type = s.try_preserve_type, verbosity = s.verbosity)
end
function MMI.transform(s::BorderlineSMOTE1, _, X::AbstractMatrix{<:Real}, y)
    borderline_smote1(X, y; m = s.m, k = s.k, ratios = s.ratios, rng = s.rng, verbosity = s.verbosity)
end


MMI.metadata_pkg(
    BorderlineSMOTE1,
    name = "Imbalance",
    package_uuid = "c709b415-507b-45b7-9a3d-1767c89fde68",
    package_url = "https://github.com/JuliaAI/Imbalance.jl",
    is_pure_julia = true,
)

MMI.metadata_model(
    BorderlineSMOTE1,
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
    load_path = "Imbalance.MLJ.BorderlineSMOTE1"
)
function MMI.transform_scitype(s::BorderlineSMOTE1)
    return Tuple{
        Union{Table(Continuous),AbstractMatrix{Continuous}},
        AbstractVector{<:Finite},
    }
end



"""
$(MMI.doc_header(BorderlineSMOTE1))

`BorderlineSMOTE1` implements the BorderlineSMOTE1 algorithm to correct for class imbalance as in
Han, H., Wang, W.-Y., & Mao, B.-H. (2005). Borderline-SMOTE: A new over-sampling method in imbalanced data sets learning. 
In D.S. Huang, X.-P. Zhang, & G.-B. Huang (Eds.), Advances in Intelligent Computing (pp. 878-887). Springer. 


# Training data

In MLJ or MLJBase, wrap the model in a machine by

    mach = machine(model)

There is no need to provide any data here because the model is a static transformer.

Likewise, there is no need to `fit!(mach)`.

For default values of the hyper-parameters, model can be constructed by

    model = BorderlineSMOTE1()


# Hyperparameters

- `m::Integer=5`: The number of neighbors to consider while checking the BorderlineSMOTE1 condition. Should be within the range 
   `0 < m < N` where N is the number of observations in the data. It will be automatically set to `N-1` if `N ≤ m`.

- `k::Integer=5`: Number of nearest neighbors to consider in the SMOTE part of the algorithm. Should be within the range
   `0 < k < n` where n is the number of observations in the smallest class. It will be automatically set to
   `l-1` for any class with `l` points where `l ≤ k`.

$((COMMON_DOCS["RATIOS"]))

$((COMMON_DOCS["RNG"]))

- `verbosity::Integer=1`: Whenever higher than `0` info regarding the points that will participate in oversampling is logged.


# Transform Inputs

$((COMMON_DOCS["INPUTS"]))

# Transform Outputs

$((COMMON_DOCS["OUTPUTS"]))

# Operations

- `transform(mach, X, y)`: resample the data `X` and `y` using BorderlineSMOTE1, returning both the
  new and original observations


# Example

```julia
using MLJ
import Imbalance

# set probability of each class
class_probs = [0.5, 0.2, 0.3]                         
num_rows, num_continuous_feats = 1000, 5
# generate a table and categorical vector accordingly
X, y = Imbalance.generate_imbalanced_data(num_rows, num_continuous_feats; 
                                stds=[0.1 0.1 0.1], min_sep=0.01, class_probs, rng=42)            

julia> Imbalance.checkbalance(y)
1: ▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇ 200 (40.8%) 
2: ▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇ 310 (63.3%) 
0: ▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇ 490 (100.0%) 

# load BorderlineSMOTE1
BorderlineSMOTE1 = @load BorderlineSMOTE1 pkg=Imbalance

# wrap the model in a machine
oversampler = BorderlineSMOTE1(m=3, k=5, ratios=Dict(0=>1.0, 1=> 0.9, 2=>0.8), rng=42)
mach = machine(oversampler)

# provide the data to transform (there is nothing to fit)
Xover, yover = transform(mach, X, y)


julia> Imbalance.checkbalance(yover)
2: ▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇ 392 (80.0%) 
1: ▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇ 441 (90.0%) 
0: ▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇ 490 (100.0%) 
```

"""
BorderlineSMOTE1