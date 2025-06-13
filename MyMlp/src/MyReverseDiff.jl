module MyReverseDiff
export topological_sort, forward!, backward!, reset!, update!, compute!,
       binarycrossentropy, dense3D, relu, transpose, σ, *, +, conv, max_pool, flatten, AdamState, setup_optimizer,
       show, summary
export Constant, Variable, ScalarOperator, BroadcastedOperator, GraphNode, Operator

using Statistics
using Distributions


import Base: *, +, clamp, log, exp
import LinearAlgebra: mul!
import Statistics: sum


abstract type GraphNode end
abstract type Operator <: GraphNode end

mutable struct Constant{T} <: GraphNode
    output :: T
end

mutable struct Variable{T<:AbstractArray{Float32}} <: GraphNode
    output :: T
    gradient :: T
    name :: String
    
    Variable(output::T; name="?") where {T<:AbstractArray{Float32}} = new{T}(output, zeros(Float32, size(output)), name)
end

mutable struct ScalarOperator{F} <: Operator
    inputs :: Tuple{GraphNode, GraphNode}
    output :: Float32
    gradient :: Float32
    name :: String
    ScalarOperator(fun, inputs...; name="?") = new{typeof(fun)}(inputs, 0.0f0, 0.0f0, name)
end

mutable struct BroadcastedOperator{F} <: Operator
    inputs :: NTuple{N, GraphNode} where N
    output :: AbstractArray{Float32}
    gradient :: AbstractArray{Float32}
    name :: String
    BroadcastedOperator(fun, inputs...; name="?") = new{typeof(fun)}(inputs, zeros(Float32, 1, 1), zeros(Float32, 1, 1), name)
end

function visit(node::GraphNode, visited, order)
    if node ∈ visited
    else
        push!(visited, node)
        push!(order, node)
    end
    return nothing
end

function visit(node::Operator, visited, order)
    if node ∈ visited
    else
        push!(visited, node)
        for input in node.inputs
            visit(input, visited, order)
        end
        push!(order, node)
    end
    return nothing
end

function topological_sort(head::GraphNode)
    visited = Set{GraphNode}()
    order = Vector{GraphNode}()
    visit(head, visited, order)
    return order
end

# x * y (aka matrix multiplication)
*(A::GraphNode, x::GraphNode; name="mul") = BroadcastedOperator(mul!, A, x, name=name)
forward(::BroadcastedOperator{typeof(mul!)}, A, x) = return A * x
backward(::BroadcastedOperator{typeof(mul!)}, A, x, g) = tuple(g * x', A' * g)

# relu activation
relu(x::GraphNode; name="relu") = BroadcastedOperator(relu, x, name=name)
forward(::BroadcastedOperator{typeof(relu)}, x) = return x .* (x .> 0.0f0)
backward(::BroadcastedOperator{typeof(relu)}, x, g) = tuple(g .* (x .> 0.0f0))

# add operation (for bias)
+(x::GraphNode, y::GraphNode; name="sum") = BroadcastedOperator(+, x, y, name=name)
forward(::BroadcastedOperator{typeof(+)}, x, y) = begin
    return x .+ y
end
backward(::BroadcastedOperator{typeof(+)}, x, y, g::Array{Float32,2}) = begin
    grad_wrt_x = g
    grad_wrt_y = sum(g, dims=2)
    return (grad_wrt_x, grad_wrt_y)
end
backward(::BroadcastedOperator{typeof(+)}, x, y, g::Array{Float32,3}) = begin
    grad_wrt_x = g
    grad_wrt_y = sum(sum(g, dims=1),dims=3)
    return (grad_wrt_x, grad_wrt_y)
end

# sigmoid activation
σ(x::GraphNode; name="sigmoid") = BroadcastedOperator(σ, x, name=name)
forward(::BroadcastedOperator{typeof(σ)}, x) = return 1.0f0 ./ (1.0f0 .+ exp.(-x))
backward(node::BroadcastedOperator{typeof(σ)}, x, g) = begin
    y = node.output
    local_derivative = y .* (1.0f0 .- y)
    grad_wrt_x = g .* local_derivative
    return (grad_wrt_x, )
end

# transpose operations
transpose(x::GraphNode; name="Transposition") = BroadcastedOperator(transpose, x, name=name)
forward(::BroadcastedOperator{typeof(transpose)},x::Matrix{Float32}) = return permutedims(x, (2,1))
forward(::BroadcastedOperator{typeof(transpose)},x::Array{Float32,3}) = return permutedims(x, (2,1,3))
backward(::BroadcastedOperator{typeof(transpose)},x,g::Matrix{Float32}) = return (permutedims(g, (2,1)),)
backward(::BroadcastedOperator{typeof(transpose)},x,g::Array{Float32,3}) = return (permutedims(g, (2,1,3)),)


# Binary Cross Entropy
function binary_cross_entropy_loss_impl(ŷ, y_true; epsilon=1e-10)
    ŷ_clamped = clamp.(ŷ, epsilon, 1.0f0 - epsilon)
    loss_elements = -y_true .* log.(ŷ_clamped) .- (1.0f0 .- y_true) .* log.(1.0f0 .- ŷ_clamped)
    return Float32(mean(loss_elements))
end

binarycrossentropy(ŷ::GraphNode, y::GraphNode; name="bce_loss") = ScalarOperator(binary_cross_entropy_loss_impl, ŷ, y, name=name)

forward(::ScalarOperator{typeof(binary_cross_entropy_loss_impl)}, ŷ_value, y_value) = begin
    loss_value = binary_cross_entropy_loss_impl(ŷ_value, y_value)
    return loss_value
end

backward(::ScalarOperator{typeof(binary_cross_entropy_loss_impl)}, ŷ_value, y_value, g) = begin
    epsilon = 1e-10
    ŷ_clamped_for_grad = clamp.(ŷ_value, epsilon, 1.0f0 - epsilon)
    local_grad_per_sample = (ŷ_clamped_for_grad .- y_value) ./ (ŷ_clamped_for_grad .* (1.0f0 .- ŷ_clamped_for_grad))
    grad_wrt_ŷ = local_grad_per_sample .* g
    return (grad_wrt_ŷ, zeros(Float32, size(y_value)))
end


# Convolution
@inline function im2col(x, k)
    steps = size(x,1) - k + 1
    B = Array{Float32, 3}(undef, steps, size(x,2)*k ,size(x,3))
    for i in 1:steps
        @views B[i, :, :] = reshape(x[i:(i + k - 1), :, :], 1, :, size(x,3)) 
    end
    return B
end

function convolution(x::Array{Float32,3},m::Array{Float32,3})
    x_new = Array{Float32, 3}(undef, size(x,1) - size(m,1) + 1, (size(x,2) - size(m,2) + 1)*size(m,3), size(x,3))
    kernel = reshape(m, :, size(m,3))
    data = im2col(x, size(m,1))
    for i in 1:size(x,3)
        @views x_new[:,:,i] .= data[:,:,i] * kernel
    end
    return x_new
end

function dif_convolution(x::Array{Float32,3}, m::Array{Float32,3}, g::Array{Float32,3})
    dx = zeros(Float32, size(x))
    dm = zeros(Float32, size(m))
    
    x_cols = im2col(x, size(m,1))

    for i in 1:size(x,3)
        dm .+= reshape(x_cols[:,:,i]' * g[:,:,i], size(m))
    end

    rm = reshape(permutedims(reverse(m, dims=1), (1,3,2)), :, size(m,2))

    pad = size(m,1) - 1
    g_padded = zeros(Float32, size(g,1) + 2 * pad, size(g,2), size(x,3))
    g_padded[pad+1:end-pad, :, :] .= g

    g_cols = im2col(g_padded, size(m,1))

    for i in 1:size(x,3)
        dx[:,:,i] .= g_cols[:,:,i] * rm
    end

    #reverse!(dm,dims=1)
    return dx, dm
end



# Max Pool
function m_pool(x::Array{Float32,3},mf::Matrix{Float32})
    m = Int64(mf[1])
    x_new = zeros(Float32,div(size(x,1),m),size(x,2),size(x,3))
    for z=1:size(x_new,3)
        for i=1:size(x_new,2)
            for j=1:size(x_new,1)
                @views a = argmax(x[j*m-m+1:j*m,i,z])
                x_new[j,i,z] = x[j*m-m+a,i,z]
            end
        end
    end
    return x_new
end

function dif_max_pool(x::Array{Float32,3},mf::Matrix{Float32}, g::Array{Float32,3})
    m = Int64(mf[1])
    x_new = zero(x)
    for z=1:size(x_new,3)
        for i=1:size(x_new,2)
            for j=1:div.(size(x_new,1),m)
                @views a = argmax(x[j*m-m+1:j*m, i,z])
                x_new[a+(j-1)*m,i,z] = g[j,i,z]
            end
        end
    end
    return tuple(x_new, 1.0f0)
end

#CNN
conv(x::GraphNode, m::GraphNode) = BroadcastedOperator(conv,x,m)
forward(::BroadcastedOperator{typeof(conv)}, x, m) = return convolution(x,m)
backward(::BroadcastedOperator{typeof(conv)}, x, m, g) = return dif_convolution(x,m,g)

max_pool(x::GraphNode, m::GraphNode) = BroadcastedOperator(max_pool,x,m)
forward(::BroadcastedOperator{typeof(max_pool)}, x, m) = return m_pool(x, m)
backward(::BroadcastedOperator{typeof(max_pool)}, x, m, g) = return dif_max_pool(x,m,g)

flatten(x::GraphNode) = BroadcastedOperator(flatten, x)
forward(::BroadcastedOperator{typeof(flatten)}, x) = return reshape(x,size(x,1)*size(x,2),size(x,3))
backward(::BroadcastedOperator{typeof(flatten)}, x, g) = begin
    return (reshape(g,size(x,1),size(x,2),size(x,3)),)
end

reset!(node::Constant) = nothing
reset!(node::Variable) = node.gradient = zeros(Float32, size(node.output))

function reset!(node::Operator)
    if isa(node.output, AbstractArray{Float32})
        node.gradient = zeros(Float32, size(node.output))
    else
        node.gradient = 0.0f0
    end
end

compute!(node::Constant) = nothing
compute!(node::Variable) = nothing


function compute!(node::Operator)
    new_output_val = forward(node, [input.output for input in node.inputs]...)

    if isa(node, ScalarOperator)
        node.output = Float32(new_output_val)
        node.gradient = 0.0f0
    else
        if size(node.output) != size(new_output_val)
            node.output = new_output_val
        else
            copyto!(node.output, Float32.(new_output_val))
        end
        if size(node.gradient) != size(node.output)
            node.gradient = zeros(Float32, size(node.output))
        else
            fill!(node.gradient, 0.0f0)
        end
    end
end

function forward!(order::Vector)
    for node in order
        compute!(node)
        reset!(node)
    end
    return last(order).output
end


update!(node::Constant, gradient) = nothing

update!(node::GraphNode, gradient) = let
    if isnothing(node.gradient)
        node.gradient = gradient 
    else
        node.gradient .+= gradient
    end
end

function backward!(order::Vector; seed=1.0f0)
    result = last(order)
    if all(iszero, result.gradient)
        if isa(result.output, AbstractArray{Float32})
            result.gradient = ones(Float32, size(result.output))
        else
            result.gradient = seed
            @assert length(result.output) == 1 "Gradient is defined only for scalar functions"
        end
    end
    for node in reverse(order)
        backward!(node)
    end
    return nothing
end

function backward!(node::Constant) end
function backward!(node::Variable) end

function backward!(node::Operator)
    inputs = node.inputs

    gradients = backward(node, [input.output for input in inputs]..., node.gradient)

    for (input, gradient) in zip(inputs, gradients)
        update!(input, gradient)
    end
    return nothing
end


import Base: show, summary
show(io::IO, x::ScalarOperator{F}) where {F} = print(io, "op ", x.name, "(", F, ")");
show(io::IO, x::BroadcastedOperator{F}) where {F} = print(io, "op.", x.name, "(", F, ")");
show(io::IO, x::Constant) = print(io, "const ", x.output)
show(io::IO, x::Variable) = begin
    print(io, "var ", x.name);
    print(io, "\n ┣━ ^ "); summary(io, x.output)
    print(io, "\n ┗━ ∇ ");  summary(io, x.gradient)
end

# Include submodule
include("MyEmbedding.jl")
using .MyEmbedding

# Re-export embedding functions
export embedding

end