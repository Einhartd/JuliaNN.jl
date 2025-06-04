include("../MyReverseDiff.jl")
include("../MyEmbedding.jl")
include("../MyMlp.jl")
include("../TensorOperations.jl")

using .MyReverseDiff
using .MyMlp
using JLD2
using Printf
using Random

x = randn(Float32,50,130,64)
m = randn(Float32,3,1)
g = ones(Float32,size(x,1),size(x,2)*size(m,2),size(x,3))

function xd(x,m,g; iter=100)
    for i=1:iter
        MyReverseDiff.dif_convolution(x,m,g)
    end
    return nothing
end

using BenchmarkTools
#@btime MyReverseDiff.dif_convolution(x,m,g)
@profview xd(x,m,g)