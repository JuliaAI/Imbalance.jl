using Test
using CategoricalArrays
using DataFrames
using Random
using NearestNeighbors
using MLUtils


include("test_utils.jl")

@testset "SMOTE" begin include("smote.jl") end
@testset "Utils" begin include("utils.jl") end