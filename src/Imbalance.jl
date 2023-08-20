"""
Entry points to the package.
"""

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
using CategoricalArrays

const MMI = MLJModelInterface

include("commondocs.jl")
include("errors.jl")
include("table_wrappers.jl")
include("generic_oversample.jl")
include("utils.jl")
export generate_imbalanced_data

include("class_counts.jl")

include("random_oversample/random_oversample.jl")
include("random_oversample/interface_mlj.jl")
include("random_oversample/interface_tables.jl")
export random_oversample, RandomOversampler, RandomOversampler_t

include("rose/rose.jl")
include("rose/interface_mlj.jl")
include("rose/interface_tables.jl")
export rose, ROSE, TableTransforms

include("common_smote.jl")
include("smote/smote.jl")
include("smote/interface_mlj.jl")
include("smote/interface_tables.jl")
export smote, SMOTE, SMOTE_t

include("smotenc/smotenc.jl")
include("smotenc/interface_mlj.jl")
include("smotenc/interface_tables.jl")
export smotenc, SMOTENC, SMOTENC_t

include("smoten/smoten.jl")
include("smoten/interface_mlj.jl")
include("smoten/interface_tables.jl")
export smoten, SMOTEN, SMOTEN_t


include("mlj_interface.jl")
end
