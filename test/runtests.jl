using Test
using Imbalance
using CategoricalArrays
using DataFrames
using Random
using MLJTestInterface
using Statistics
using Clustering
using Distances
using StatsBase: countmap, mode, proportions
using NearestNeighbors, Distances
using MLJBase: machine, transform
using StableRNGs: StableRNG
using TableTransforms
using TransformsBase
using ScientificTypes
using Serialization
using IOCapture
using JLD2
ENV["PYTHON"] = ""
using Pkg: Pkg;
Pkg.build("PyCall");
using PyCall
using Conda

include("test_utils.jl")
# When the following variable is set to false, offline results from PyCall will be used
# When it is set to true, PyCall will be used instead of saved resules and it will save the results.

offline_python_test = true

### general

@testset "class_counts" begin
	include("class_counts.jl")
end

@testset "table_wrappers" begin
	include("table_wrappers.jl")
end

@testset "generic_resample" begin
	include("generic_resample.jl")
end

@testset "extras" begin
	include("extras.jl")
end


@testset "distance metrics" begin
	include("distance_metrics.jl")
end



### interfaces

@testset "MLJ Interface" begin
	include("interfaces/mlj_interface.jl")
end

@testset "tabletransforms" begin
	include("interfaces/tabletr_interface.jl")
end


### undersampling

@testset "Basic Random Undersampler" begin
	include("undersampling/random_undersample.jl")
end

@testset "ENN Undersampler" begin
	include("undersampling/enn_undersample.jl")
end

@testset "Tomek Undersampler" begin
	include("undersampling/tomek_undersample.jl")
end


@testset "Cluster Undersampler" begin
	include("undersampling/cluster_undersample.jl")
end



### oversampling

@testset "Basic Random Oversampler" begin
	include("oversampling/random_oversample.jl")
end

@testset "Random Walk Oversampling" begin
	include("oversampling/random_walk.jl")
end

@testset "ROSE" begin
	include("oversampling/rose.jl")
end

@testset "SMOTE" begin
	include("oversampling/smote.jl")
end

@testset "BorderlineSMOTE1" begin
	include("oversampling/borderline_smote1.jl")
end


@testset "SMOTEN" begin
	include("oversampling/smoten.jl")
end

@testset "SMOTENC" begin
	include("oversampling/smotenc.jl")
end
