using Test
using Imbalance
using CategoricalArrays
using DataFrames
using Random
using MLJTestInterface
using Statistics
using Clustering
using Distances
using StatsBase: countmap, mode
using NearestNeighbors, Distances
using MLJBase: machine, transform
using StableRNGs: StableRNG
using TableTransforms
using TransformsBase
using ScientificTypes
using IOCapture
ENV["PYTHON"]=""
import Pkg; Pkg.build("PyCall")
using PyCall       
using Conda                 



include("test_utils.jl")
@testset "common_utils" begin
    include("common_utils.jl")
end

@testset "table_wrappers" begin
    include("table_wrappers.jl")
end

@testset "generic_resample" begin
    include("generic_resample.jl")
end

@testset "tabletransforms" begin
    include("tabletr_interface.jl")
end
@testset "MLJ Interface" begin
    include("mlj_interface.jl")
end

@testset "distance metrics" begin
    include("distance_metrics.jl")
end

@testset "ROSE" begin
    include("rose.jl")
end

@testset "Basic Random Oversampler" begin
    include("random_oversample.jl")
end

@testset "Basic Random Undersampler" begin
    include("random_undersample.jl")
end

@testset "ENN Undersampler" begin
    include("enn_undersample.jl")
end

@testset "Tomek Undersampler" begin
    include("tomek_undersample.jl")
end


@testset "Cluster Undersampler" begin
    include("cluster_undersample.jl")
end

@testset "SMOTE" begin
    include("smote.jl")
end

@testset "SMOTENC" begin
    include("smotenc.jl")
end


@testset "SMOTEN" begin
    include("smoten.jl")
end
