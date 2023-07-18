using Test
using CategoricalArrays
using DataFrames
using Random
using NearestNeighbors


include("test_utils.jl")

@testset "common" begin include("common.jl") end
@testset "Basic Random Oversampler" begin include("basic.jl") end
@testset "SMOTE" begin include("smote.jl") end
@testset "ROSE" begin include("rose.jl") end
