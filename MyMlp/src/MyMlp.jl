module MyMlp
using BenchmarkTools
using LinearAlgebra
using MLDatasets
using Plots

# Import ReverseDiff module
include("MyReverseDiff.jl")
export MyReverseDiff

export greet

greet() = print("Hello World!")


end # module MyMlp

