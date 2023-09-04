"""
Entry points to the package.
"""

module Imbalance

using Random: AbstractRNG, default_rng, shuffle, Xoshiro
using Statistics
using StatsBase: mode, countmap
using LinearAlgebra
using ScientificTypes
using NearestNeighbors, Distances
using CategoricalDistributions
using TableOperations
using Tables
using OrderedCollections
using CategoricalArrays
using ProgressMeter
using Memoization


include("commondocs.jl")
include("errors.jl")
include("table_wrappers.jl")
include("generic_oversample.jl")
include("generic_encoding.jl")
include("common_smote.jl")
include("utils.jl")
export generate_imbalanced_data, checkbalance

include("class_counts.jl")
include("random_oversample/random_oversample.jl")
include("rose/rose.jl")
include("smote/smote.jl")
include("smoten/smoten.jl")
include("smotenc/smotenc.jl")
export random_oversample, rose, smote, smoten, smotenc


module MLJ
	using Random: AbstractRNG, default_rng
	using MLJModelInterface
	const MMI = MLJModelInterface
	using ..Imbalance: random_oversample, rose, smote, smoten, 
                       smotenc, COMMON_DOCS
	include("random_oversample/interface_mlj.jl")
	include("rose/interface_mlj.jl")
	include("smote/interface_mlj.jl")
	include("smotenc/interface_mlj.jl")
	include("smoten/interface_mlj.jl")
	export RandomOversampler, ROSE, SMOTE, SMOTEN, SMOTENC
end

module TableTransforms
	using Random: AbstractRNG, default_rng
	using TransformsBase
	using ..Imbalance: random_oversample, rose, smote, smoten, 
                       smotenc, COMMON_DOCS, rowcount, revert_oversampling
	include("random_oversample/interface_tables.jl")
	include("rose/interface_tables.jl")
	include("smote/interface_tables.jl")
	include("smotenc/interface_tables.jl")
	include("smoten/interface_tables.jl")
	export RandomOversampler, ROSE, SMOTE, SMOTEN, SMOTENC
end


end
