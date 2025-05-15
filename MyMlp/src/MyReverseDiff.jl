module MyReverseDiff
export Variable

# Now try importing
using BenchmarkTools
using LinearAlgebra

import Base: *, +, clamp, log, exp
import LinearAlgebra: mul!
import Statistics: sum

abstract type GraphNode end
abstract type Operator <: GraphNode end

# Definition of basic structures for computational graph
struct Constant{T<:AbstractVecOrMat{Float64}} <: GraphNode
    output :: T
end

mutable struct Variable{T<:AbstractVecOrMat{Float64}} <: GraphNode
    output :: T
    gradient :: Union{Nothing, T}
    name :: String
    
    Variable(output::T; name="?") where {T<:AbstractVecOrMat{Float64}} = new{T}(output, nothing, name)
end

mutable struct ScalarOperator{F} <: Operator
    inputs :: Union{Nothing, Tuple{GraphNode, GraphNode}}
    output :: Union{Nothing, Float64}
    gradient :: Union{Nothing, Float64}
    name :: String
    ScalarOperator(fun, inputs...; name="?") = new{typeof(fun)}(inputs, nothing, nothing, name)
end

mutable struct BroadcastedOperator{F} <: Operator
    inputs :: Union{Nothing, Tuple{GraphNode, GraphNode}, Tuple{GraphNode}}
    output :: Union{Nothing, AbstractVecOrMat{Float64}}
    gradient :: Union{Nothing, Matrix{Float64}}
    name :: String
    BroadcastedOperator(fun, inputs...; name="?") = new{typeof(fun)}(inputs, nothing, nothing, name)
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
    visited = Set()
    order = Vector()
    visit(head, visited, order)
    return order
end


# x * y (aka matrix multiplication)
*(A::GraphNode, x::GraphNode) = BroadcastedOperator(mul!, A, x)
forward(::BroadcastedOperator{typeof(mul!)}, A, x) = return A * x
backward(::BroadcastedOperator{typeof(mul!)}, A, x, g) = tuple(g * x', A' * g)

# relu activation
relu(x::GraphNode) = BroadcastedOperator(relu, x)
forward(::BroadcastedOperator{typeof(relu)}, x) = return x .* (x .> 0.0)
backward(::BroadcastedOperator{typeof(relu)}, x, g) = tuple(g .* (x .> 0.0), nothing)

# add operation (for bias)
+(x::GraphNode, y::GraphNode) = BroadcastedOperator(+, x, y)
forward(::BroadcastedOperator{typeof(+)}, x, y) = return x .+ y
backward(::BroadcastedOperator{typeof(+)}, x, y, g) = begin
    grad_wrt_x = g
    # Gradient biasu: sumowanie gradientu g wzdłuż wymiaru batcha (dims=1)
    # i przekształcenie wyniku 1xN do wektora N-elementowego.
    grad_wrt_y = vec(sum(g, dims=2))
    return (grad_wrt_x, grad_wrt_y)
end

# sigmoid activation
σ(x::GraphNode) = BroadcastedOperator(σ, x)
forward(::BroadcastedOperator{typeof(σ)}, x) = return 1.0 ./ (1.0 .+ exp.(-x))
backward(node::BroadcastedOperator{typeof(σ)}, x, g) = begin
    y = node.output
    local_derivative = y .* (1.0 .- y)
    grad_wrt_x = g .* local_derivative
    return (grad_wrt_x, nothing)
end

function binary_cross_entropy_loss_impl(ŷ, y_true; epsilon=1e-10)
    ŷ_clamped = clamp.(ŷ, epsilon, 1.0 - epsilon)
    loss_elements = -y_true .* log.(ŷ_clamped) .- (1.0 .- y_true) .* log.(1.0 .- ŷ_clamped)
    return mean(loss_elements)
end

binarycrossentropy(ŷ::GraphNode, y::GraphNode) = ScalarOperator(binary_cross_entropy_loss_impl, ŷ, y)

forward(::ScalarOperator{typeof(binary_cross_entropy_loss_impl)}, ŷ_value, y_value) = begin
    loss_value = binary_cross_entropy_loss_impl(ŷ_value, y_value)
    return loss_value
end

backward(::ScalarOperator{typeof(binary_cross_entropy_loss_impl)}, ŷ_value, y_value, g) = begin
    epsilon = 1e-10
    ŷ_clamped_for_grad = clamp.(ŷ_value, epsilon, 1.0 - epsilon)
    local_grad_per_sample = (ŷ_clamped_for_grad .- y_value) ./ (ŷ_clamped_for_grad .* (1.0 .- ŷ_clamped_for_grad))
    batch_size = size(y_value, 2)
    grad_wrt_ŷ = local_grad_per_sample ./ batch_size
    return (grad_wrt_ŷ, nothing)
end

reset!(node::Constant) = nothing

reset!(node::Variable) = node.gradient = nothing
reset!(node::Operator) = node.gradient = nothing

compute!(node::Constant) = nothing
compute!(node::Variable) = nothing

compute!(node::Operator) =
    node.output = forward(node, [input.output for input in node.inputs]...)

function forward!(order::Vector)
    #   Iteruje przez każdy węzeł w order.
    for node in order
        compute!(node)
        reset!(node)
    end
    return last(order).output
end

update!(node::Constant, gradient) = nothing

update!(node::GraphNode, gradient) = if isnothing(node.gradient)
    node.gradient = gradient else node.gradient .+= gradient
end


function backward!(order::Vector; seed=1.0)
    result = last(order)   #   The output node

    if isa(result.output, Matrix{Float64})
        result.gradient = ones(Float64, size(result.output))
    else
        result.gradient = seed
        @assert length(result.output) == 1 "Gradient is defined only for scalar functions"
    end

    for node in reverse(order)   #   Iterate through nodes in reverse topological order.
        backward!(node)   #   Compute and propagate gradients backwards.
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

end