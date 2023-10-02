
### BorderlineSMOTE1 with MLJ Interface

mutable struct BorderlineSMOTE1{T,R<:Union{Integer,AbstractRNG}, I<:Integer} <: Static
    m::I
    k::I
    ratios::T
    rng::R
    try_perserve_type::Bool
    verbosity::I
end;



"""
Check whether the given model hyperparameters are valid and clean them if necessary. 
"""
function MMI.clean!(s::BorderlineSMOTE1)
    message = ""
    if s.k < 1
        throw(ERR_NONPOS_K(s.k))
    end
    if s.m < 1
        throw(ERR_NONPOS_K(s.m))
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
    rng::Union{Integer,AbstractRNG} = default_rng(), try_perserve_type::Bool=true, verbosity::Integer=1
) where {T}
    model = BorderlineSMOTE1(m, k, ratios, rng, try_perserve_type, verbosity)
    MMI.clean!(model)
    return model
end

"""
Oversample data X, y using BorderlineSMOTE1
"""
function MMI.transform(s::BorderlineSMOTE1, _, X, y)
    borderline_smote1(X, y; m = s.m, k = s.k, ratios = s.ratios, rng = s.rng, 
        try_perserve_type = s.try_perserve_type, verbosity = s.verbosity)
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
    input_scitype = Union{Table(Continuous),AbstractMatrix{Continuous}},
    output_scitype = Union{Table(Continuous),AbstractMatrix{Continuous}},
    target_scitype = AbstractVector,
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
    `n-1` for any class where `n ≤ k`.

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

```
using MLJ
import Random.seed!
using MLUtils
import StatsBase.countmap

seed!(12345)

# Generate some imbalanced data:
X, y = @load_iris # a table and a vector
rand_inds = rand(1:150, 30)
X, y = getobs(X, rand_inds), y[rand_inds]

julia> countmap(y)
Dict{CategoricalArrays.CategoricalValue{String, UInt32}, Int64} with 3 entries:
  "virginica"  => 12
  "versicolor" => 5
  "setosa"     => 13

# load BorderlineSMOTE1 model type:
BorderlineSMOTE1 = @load BorderlineSMOTE1 pkg=Imbalance

# Oversample the minority classes to  sizes relative to the majority class:
oversampler = BorderlineSMOTE1(k=10, ratios=Dict("setosa"=>1.0, "versicolor"=> 0.8, "virginica"=>1.0), rng=42)
mach = machine(oversampler)
Xover, yover = transform(mach, X, y)

julia> countmap(yover)
Dict{CategoricalArrays.CategoricalValue{String, UInt32}, Int64} with 3 entries:
  "virginica"  => 13
  "versicolor" => 10
  "setosa"     => 13
```

"""
BorderlineSMOTE1
