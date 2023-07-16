using Test
using Plots
using CategoricalArrays
using DataFrames

include("test_utils.jl")

@testset "smote" begin include("smote.jl") end