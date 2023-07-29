module Imbalance

using Random: AbstractRNG, default_rng
using StableRNGs: StableRNG
using Statistics
using LinearAlgebra
using NearestNeighbors
using Tables
using MLJModelInterface
using OrderedCollections
using Parameters
const MMI = MLJModelInterface

include("commondocs.jl")
include("common.jl")
include("utils.jl")

include("class_counts.jl")

include("basic/basic.jl")
include("basic/interfaces.jl")
export random_oversample

include("rose/rose.jl")
include("rose/interfaces.jl")
export rose

include("smote/smote.jl")
include("smote/interfaces.jl")
export smote


include("mlj_interface.jl")
export SMOTE, ROSE, RandomOversampler
end