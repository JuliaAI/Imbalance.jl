"""
Entry points to the package.
"""

module Imbalance

using Random: AbstractRNG, default_rng, shuffle, Xoshiro, seed!
using Statistics
using StatsBase: mode, countmap, sample
using LinearAlgebra
using ScientificTypes
using NearestNeighbors, Distances
using Clustering
using CategoricalDistributions
using TableOperations
using Tables
using OrderedCollections
using CategoricalArrays
using ProgressMeter


include("common/docs.jl")
include("common/errors.jl")
include("common/utils.jl")

include("table_wrappers.jl")
include("generic_resample.jl")
include("generic_encoding.jl")

include("extras.jl")
export generate_imbalanced_data, checkbalance

include("class_counts.jl")
include("random_oversample/random_oversample.jl")
include("rose/rose.jl")
include("smote/smote.jl")
include("smoten/smoten.jl")
include("smotenc/smotenc.jl")
export random_oversample, rose, smote, smoten, smotenc

include("random_undersample/random_undersample.jl")
include("cluster_undersample/cluster_undersample.jl")
export random_undersample, cluster_undersample

module MLJ
	using Random: AbstractRNG, default_rng
	using MLJModelInterface
	const MMI = MLJModelInterface
	using ..Imbalance: random_oversample, rose, smote, smoten, 
                       smotenc, random_undersample, COMMON_DOCS
	include("random_oversample/interface_mlj.jl")
	include("rose/interface_mlj.jl")
	include("smote/interface_mlj.jl")
	include("smotenc/interface_mlj.jl")
	include("smoten/interface_mlj.jl")
	include("random_undersample/interface_mlj.jl")
	export RandomOversampler, ROSE, SMOTE, SMOTEN, SMOTENC, RandomUndersampler
end

module TableTransforms
	using Random: AbstractRNG, default_rng
	using TransformsBase
	using ..Imbalance: random_oversample, rose, smote, smoten, 
                       smotenc, random_undersample, COMMON_DOCS, rowcount, revert_oversampling
	include("random_oversample/interface_tables.jl")
	include("rose/interface_tables.jl")
	include("smote/interface_tables.jl")
	include("smotenc/interface_tables.jl")
	include("smoten/interface_tables.jl")
	include("random_undersample/interface_tables.jl")
	export RandomOversampler, ROSE, SMOTE, SMOTEN, SMOTENC, RandomUndersampler
end


end
