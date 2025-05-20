module MyReverseDiff
export topological_sort, forward!, backward!, reset!, update!, compute!,
       binarycrossentropy, relu, σ, *, +, AdamState, setup_optimizer,
       show, summary
export Constant, Variable, ScalarOperator, BroadcastedOperator, GraphNode, Operator

using Statistics
using Distributions


import Base: *, +, clamp, log, exp
import LinearAlgebra: mul!
import Statistics: sum


# Definition of basic structures for computational graph

abstract type GraphNode end
abstract type Operator <: GraphNode end

mutable struct Constant{T<:Matrix{Float32}} <: GraphNode
    output :: T
end

mutable struct Variable{T<:Matrix{Float32}} <: GraphNode
    output :: T
    gradient :: T
    name :: String
    
    Variable(output::T; name="?") where {T<:Matrix{Float32}} = new{T}(output, zeros(Float32, size(output)), name)
end

mutable struct ScalarOperator{F} <: Operator
    inputs :: Tuple{GraphNode, GraphNode}
    output :: Float32
    gradient :: Float32
    name :: String
    ScalarOperator(fun, inputs...; name="?") = new{typeof(fun)}(inputs, 0.0f0, 0.0f0, name)
end

mutable struct BroadcastedOperator{F} <: Operator
    inputs :: Union{Tuple{GraphNode, GraphNode}, Tuple{GraphNode}}
    output :: Matrix{Float32}
    gradient :: Matrix{Float32}
    name :: String
    BroadcastedOperator(fun, inputs...; name="?") = new{typeof(fun)}(inputs, zeros(Float32, 1, 1), zeros(Float32, 1, 1), name)
end





function visit(node::GraphNode, visited, order)
    if node ∈ visited
    else
        push!(visited, node)
        push!(order, node)
    end
    return zeros(Float32, 1, 1)
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
    return zeros(Float32, 1, 1)
end

function topological_sort(head::GraphNode)
    visited = Set()
    order = Vector()
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
backward(::BroadcastedOperator{typeof(relu)}, x, g) = tuple(g .* (x .> 0.0f0), zeros(Float32, 1, 1))

# add operation (for bias)
+(x::GraphNode, y::GraphNode; name="sum") = BroadcastedOperator(+, x, y, name=name)
forward(::BroadcastedOperator{typeof(+)}, x, y) = return x .+ y
backward(::BroadcastedOperator{typeof(+)}, x, y, g) = begin
    grad_wrt_x = g
    grad_wrt_y = sum(g, dims=2)
    return (grad_wrt_x, grad_wrt_y)
end

# sigmoid activation
σ(x::GraphNode; name="sigmoid") = BroadcastedOperator(σ, x, name=name)
forward(::BroadcastedOperator{typeof(σ)}, x) = return 1.0f0 ./ (1.0f0 .+ exp.(-x))
backward(node::BroadcastedOperator{typeof(σ)}, x, g) = begin
    y = node.output
    local_derivative = y .* (1.0f0 .- y)
    grad_wrt_x = g .* local_derivative
    return (grad_wrt_x, zeros(Float32, 1, 1))
end

function binary_cross_entropy_loss_impl(ŷ, y_true; epsilon=1e-10)
    ŷ_clamped = clamp.(ŷ, epsilon, 1.0f0 - epsilon)
    loss_elements = -y_true .* log.(ŷ_clamped) .- (1.0f0 .- y_true) .* log.(1.0f0 .- ŷ_clamped)
    return mean(loss_elements)
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
    batch_size = size(y_value, 2)
    grad_wrt_ŷ = local_grad_per_sample ./ batch_size
    return (grad_wrt_ŷ, zeros(Float32, 1, 1))
end

reset!(node::Constant) = nothing
reset!(node::Variable) = node.gradient = zeros(Float32, size(node.output))

function reset!(node::Operator)
    if isa(node.output, Matrix{Float32})
        node.gradient = zeros(Float32, size(node.output))
    else
        node.gradient = 0.0f0
    end
end
#reset!(node::Operator) = node.gradient = zeros(Float32, size(node.output))

compute!(node::Constant) = nothing
compute!(node::Variable) = nothing

function compute!(node::Operator)
    node.output = forward(node, [input.output for input in node.inputs]...)
    if isa(node.output, Matrix{Float32})
        node.gradient = zeros(Float32, size(node.output))
    end
end
# compute!(node::Operator) =
#     node.output = forward(node, [input.output for input in node.inputs]...)

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
    if all(iszero, result.gradient)
        if isa(result.output, Matrix{Float32})
            result.gradient = ones(Float32, size(result.output))
        else
            result.gradient = seed
            @assert length(result.output) == 1 "Gradient is defined only for scalar functions"
        end
    end

    for node in reverse(order)   #   Iterate through nodes in reverse topological order.
        backward!(node)   #   Compute and propagate gradients backwards.
    end
    return zeros(Float32, 1, 1)
end

function backward!(node::Constant) end
function backward!(node::Variable) end

function backward!(node::Operator)
    inputs = node.inputs

    gradients = backward(node, [input.output for input in inputs]..., node.gradient)

    for (input, gradient) in zip(inputs, gradients)
        update!(input, gradient)
    end
    return zeros(Float32, 1, 1)
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


end