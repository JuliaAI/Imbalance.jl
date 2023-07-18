using Test
using CategoricalArrays
using DataFrames
using Random
using NearestNeighbors
using MLUtils


include("test_utils.jl")

@testset "Basic Random Oversampler" begin include("basic.jl") end
@testset "SMOTE" begin include("smote.jl") end
@testset "ROSE" begin include("rose.jl") end
@testset "common" begin include("common.jl") end