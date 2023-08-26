using Test
using CategoricalArrays
using DataFrames
using Random
using MLJTestInterface
using Statistics
using Distances
using StatsBase: countmap
using NearestNeighbors, Distances
using MLJBase: machine, transform
using StableRNGs: StableRNG
using TableTransforms
using ScientificTypes
import Pkg; Pkg.build("PyCall")
using PyCall       
using Conda                 



include("test_utils.jl")

@testset "common" begin
    include("common.jl")
end

@testset "wrappers" begin
    include("wrappers.jl")
end

@testset "tabletransforms" begin
    include("tabletr_interface.jl")
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


@testset "SMOTEN" begin
    include("smoten.jl")
end
