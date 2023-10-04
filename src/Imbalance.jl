"""
Entry points to the package.
"""

module Imbalance

using Random: AbstractRNG, default_rng, shuffle
using Random
using Statistics
using StatsBase: mode, countmap, sample, modes, proportions, ProbabilityWeights
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
include("common/logging.jl")
include("common/utils.jl")

include("table_wrappers.jl")
include("generic_resample.jl")
include("generic_encoding.jl")

include("extras.jl")
export generate_imbalanced_data, checkbalance

include("class_counts.jl")
include("oversampling_methods/random_oversample/random_oversample.jl")
include("oversampling_methods/rose/rose.jl")
include("oversampling_methods/smote/smote.jl")
include("oversampling_methods/borderline_smote1/borderline_smote1.jl")
include("oversampling_methods/smoten/smoten.jl")
include("oversampling_methods/smotenc/smotenc.jl")
include("oversampling_methods/random_walk/random_walk.jl")
export random_oversample, rose, smote, smoten, smotenc, random_walk_oversample, borderline_smote1

include("undersampling_methods/random_undersample/random_undersample.jl")
include("undersampling_methods/cluster_undersample/cluster_undersample.jl")
include("undersampling_methods/enn_undersample/enn_undersample.jl")
include("undersampling_methods/tomek_undersample/tomek_undersample.jl")
export random_undersample, cluster_undersample, enn_undersample, tomek_undersample

module MLJ
using Random: AbstractRNG, default_rng
using MLJModelInterface
const MMI = MLJModelInterface
using ..Imbalance:
    random_oversample,
    rose,
    smote,
    borderline_smote1,
    smoten,
    smotenc,
    random_walk_oversample,
    random_undersample,
    cluster_undersample,
    COMMON_DOCS,
    enn_undersample,
    tomek_undersample
include("oversampling_methods/random_oversample/interface_mlj.jl")
include("oversampling_methods/rose/interface_mlj.jl")
include("oversampling_methods/smote/interface_mlj.jl")
include("oversampling_methods/borderline_smote1/interface_mlj.jl")
include("oversampling_methods/smotenc/interface_mlj.jl")
include("oversampling_methods/smoten/interface_mlj.jl")
include("oversampling_methods/random_walk/interface_mlj.jl")
include("undersampling_methods/random_undersample/interface_mlj.jl")
include("undersampling_methods/cluster_undersample/interface_mlj.jl")
include("undersampling_methods/tomek_undersample/interface_mlj.jl")
include("undersampling_methods/enn_undersample/interface_mlj.jl")
export RandomOversampler,
    ROSE,
    SMOTE,
    BorderlineSMOTE1,
    SMOTEN,
    SMOTENC,
    RandomWalkOversampler,
    RandomUndersampler,
    ClusterUndersampler,
    TomekUndersampler,
    ENNUndersampler
end

module TableTransforms
using Random: AbstractRNG, default_rng
using TransformsBase
using ..Imbalance:
    random_oversample,
    rose,
    smote,
    borderline_smote1,
    smoten,
    smotenc,
    random_walk_oversample,
    random_undersample,
    cluster_undersample,
    COMMON_DOCS,
    rowcount,
    revert_oversampling,
    enn_undersample,
    tomek_undersample
include("oversampling_methods/random_oversample/interface_tables.jl")
include("oversampling_methods/rose/interface_tables.jl")
include("oversampling_methods/smote/interface_tables.jl")
include("oversampling_methods/borderline_smote1/interface_tables.jl")
include("oversampling_methods/smotenc/interface_tables.jl")
include("oversampling_methods/smoten/interface_tables.jl")
include("oversampling_methods/random_walk/interface_tables.jl")
include("undersampling_methods/random_undersample/interface_tables.jl")
include("undersampling_methods/cluster_undersample/interface_tables.jl")
include("undersampling_methods/enn_undersample/interface_tables.jl")
include("undersampling_methods/tomek_undersample/interface_tables.jl")
export RandomOversampler,
    ROSE,
    SMOTE,
    BorderlineSMOTE1,
    SMOTEN,
    SMOTENC,
    RandomWalkOversampler,
    RandomUndersampler,
    ClusterUndersampler,
    TomekUndersampler,
    ENNUndersampler
end

end
