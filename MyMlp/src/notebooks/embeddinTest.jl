import Pkg
# Pkg.add("BenchmarkTools")
# Pkg.add("LinearAlgebra")
# Pkg.add("Statistics")
# Pkg.add("Distributions")
# Pkg.add("Random")
# Pkg.add("Plots")
# Pkg.add("MLDatasets")
# Pkg.add("DataFrames")
# Pkg.add("MLDataUtils")
# Pkg.add("JLD2")
# Pkg.add("UUIDs")


include("../MyReverseDiff.jl")
include("../MyEmbedding.jl")
include("../MyMlp.jl")

using .MyReverseDiff
using .MyEmbedding
using .MyMlp
using JLD2
using Printf
using BenchmarkTools
using LinearAlgebra
using Distributions
using Random
using MLDatasets
using Plots
using Statistics
using DataFrames
using MLDataUtils


b = Float32.([1.0 3.0 5.0 0.0; 2.0 4.0 6.0 0.0])
a = Float32.([1.0 2.0; 2.0 3.0; 3.0 4.0; 4.0 4.0])

A = Constant(a)
B = Variable(b; name = "B")
embeddings = embedding(B, A; name = "embedding")
order = topological_sort(embeddings)


embedding_layer = MyMlp.Embedding(b; name="my_embedding_layer", padding_idx=4)

forward!(order)
backward!(order)