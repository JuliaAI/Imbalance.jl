module Imbalance

using Random: AbstractRNG, default_rng, shuffle, Xoshiro
using Statistics
using StatsBase: mode, countmap
using TransformsBase
using LinearAlgebra
using ScientificTypes
using Parameters
using NearestNeighbors, Distances
using CategoricalDistributions
using TableOperations
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
export random_oversample, RandomOversampler, RandomOversampler_t

include("rose/rose.jl")
include("rose/interfaces.jl")
export rose, ROSE, ROSE_t

include("common_smote.jl")
include("smote/smote.jl")
include("smote/interfaces.jl")
export smote, SMOTE, SMOTE_t

include("smotenc/smotenc.jl")
include("smotenc/interfaces.jl")
export smotenc

include("smoten/smoten.jl")
include("smoten/interfaces.jl")
export smoten


include("mlj_interface.jl")
end
