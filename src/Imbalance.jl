module Imbalance

using Random: AbstractRNG, default_rng, shuffle
using StableRNGs: StableRNG
using Statistics
using StatsBase: mode, countmap
using LinearAlgebra
using NearestNeighbors, Distances
using Tables
using MLJModelInterface
using OrderedCollections

const MMI = MLJModelInterface

include("commondocs.jl")
include("wrappers.jl")
include("utils.jl")

include("class_counts.jl")

include("basic/basic.jl")
include("basic/interfaces.jl")
export random_oversample

include("rose/rose.jl")
include("rose/interfaces.jl")
export rose

include("common_smote.jl")
include("smote/smote.jl")
include("smote/interfaces.jl")
export smote

include("smotenc/smotenc.jl")
include("smotenc/interfaces.jl")
export smotenc


include("mlj_interface.jl")
export SMOTE, ROSE, RandomOversampler
end
