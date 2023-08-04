using Test
using CategoricalArrays
using DataFrames
using Random
using MLJTestInterface
using Statistics
using StatsBase: countmap
using NearestNeighbors, Distances
using MLJBase: machine, transform
using StableRNGs: StableRNG


include("test_utils.jl")

@testset "common" begin
    include("common.jl")
end

@testset "MLJ Interface" begin
    include("mlj_interface.jl")
end

@testset "ROSE" begin
    include("rose.jl")
end

@testset "Basic Random Oversampler" begin
    include("basic.jl")
end
@testset "SMOTE" begin
    include("smote.jl")
end

@testset "SMOTENC" begin
    include("smotenc.jl")
end
