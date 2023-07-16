module Imbalance

using MLUtils
using Random: AbstractRNG, default_rng
using Statistics
using LinearAlgebra
using NearestNeighbors
using Tables


# greet function
function greet(name)
    return "Hello $name"
end


include("utils.jl")
include("smote.jl")

export SMOTE

end # module Imbalance

