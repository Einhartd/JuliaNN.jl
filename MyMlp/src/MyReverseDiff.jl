module MyReverseDiff
export topological_sort, forward!, backward!, reset!, update!, compute!,
       binarycrossentropy, relu, σ, *, +, conv, max_pool, flatten, AdamState, setup_optimizer,
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

# Convolution

function multi_convolution_fast!(x_new::Matrix{Float32},x::Matrix{Float32},m::Matrix{Float32})
    c = size(m,1)
    x_new .= reshape(im2col_p(x,c)*m,:,size(m,2)*size(x,2))
    return x_new
end

function multi_convolution_fast!(x_new::Matrix{Float32},x::SubArray{Float32, 2, Matrix{Float32}, Tuple{Base.Slice{Base.OneTo{Int64}}, StepRange{Int64, Int64}}, false},m::Matrix{Float32})
    c = size(m,1)
    x_new .= reshape(im2col_p(x,c)*m,:,size(m,2)*size(x,2))
    return x_new
end

function multi_convolution(x::Matrix{Float32},m::Matrix{Float32})
    y = zeros(Float32,size(x,1),size(x,2)*size(m,2))
    return multi_convolution_fast!(y,x,m)
end

@inline function im2col_p(Ao::Matrix{Float32}, m::Int64)
    A = zeros(Float32, size(Ao,1)+m-1, size(Ao,2))
    A[1:size(Ao,1),:] = Ao
    M,N = size(A)
    B = Array{eltype(Matrix{Float32})}(undef, m,
    (M-m+1)*(N))
    indx = reshape(1:M*N, M,N)[1:M-m+1,1:N]
    for (i,value) in enumerate(indx)
        @views B[(i-1)*m+1:(i-1)m+m] = A[value:value+m-1]
    end
    return B'
end

@inline function im2col_p(Ao::SubArray{Float32, 2, Matrix{Float32}, Tuple{Base.Slice{Base.OneTo{Int64}}, StepRange{Int64, Int64}}, false}, m::Int64)
    A = zeros(Float32, size(Ao,1)+m-1, size(Ao,2))
    A[1:size(Ao,1),:] = Ao
    M,N = size(A)
    B = Array{eltype(Matrix{Float32})}(undef, m,
    (M-m+1)*(N))
    indx = reshape(1:M*N, M,N)[1:M-m+1,1:N]
    for (i,value) in enumerate(indx)
        @views B[(i-1)*m+1:(i-1)m+m] = A[value:value+m-1]
    end
    return B'
end

function dif_convolution(x::Matrix{Float32}, m::Matrix{Float32}, g::Matrix{Float32})
    lx = size(x,1)                      #Dlugosc obrazu
    dx = zero(x)                        #Dx

    size_diff = size(m,2)               #Liczba masek
    rm = reverse(m, dims=2)
    tmp_x = zeros(Float32,size(x,1),size_diff*size_diff)
    gv = zeros(Float32,size(x,1),size_diff)
    for i=1:size(dx,2)                 #Obrazy wyjściowe
        #Zwężenie                       #obraz g,   maska
        #@views dx[:,i] = sum(multi_convolution_fast!(tmp_x, g[:,1+(i-1)*size_diff:i*size_diff], rm), dims=2)
        gv .= g[:,i:size(x,2):end]
        @views dx[:,i] = sum(multi_convolution_fast!(tmp_x, gv, rm), dims=2)
    end

    tmp_x = zeros(Float32,size(x,1),size(x,2)*size(g,2))
    multi_convolution_fast!(tmp_x,x,g)
    dm = zero(m)
    for i=1:size_diff                 #liczba masek
        t = @view(tmp_x[:, 1+(i-1)*size(x,2):i*size(x,2)])    
        tt = sum(t,dims=2)
        dm[:,i] = reverse(@view(tt[(lx-size(m,1) +1):lx,:]), dims=1)
    end
    return tuple(dx,dm)
end

function dif_convolution_fix(x::Matrix{Float32}, m::Matrix{Float32}, g::Matrix{Float32})
    dx = zero(x)                        #Dx

    size_diff = size(m,2)               #Liczba masek

    tmp_x = zeros(Float32,size(x,1),size_diff*size(g,2))
    rg = reverse(g, dims=1)
    multi_convolution_fast!(tmp_x, rg, m)
    reverse!(tmp_x,dims=1)

    for i=1:size(dx,2)                 #Obrazy wejściowe
        @views dx[:,i] = sum(tmp_x[:,((i-1)*size_diff+1):(1+size(g,2)):end], dims=2)
    end

    tmp_x = zeros(Float32,size(x,1),size(x,2)*size(g,2))
    multi_convolution_fast!(tmp_x,x,g)
    dm = zero(m)
    for i=1:size_diff                 #liczba masek
        tt = sum(tmp_x[:, i:(size(g,2)+size_diff):end],dims=2)
        @views dm[:,i] = tt[1:size(m,1),:]
    end
    return tuple(dx,dm)
end

# Max Pool
function m_pool(x::Matrix{Float32},mf::Matrix{Float32})
    m = Int64(mf[1])
    x_new = zeros(Float32,div(size(x,1),m),size(x,2))
    for i=1:size(x_new,2)
        for j=1:size(x_new,1)
            a = argmax(x[j*m-m+1:j*m,i])
            x_new[j,i] = x[j*m-m+a,i]
        end
    end
    return x_new
end

function dif_max_pool(x::Matrix{Float32},mf::Matrix{Float32}, g::Matrix{Float32})
    m = Int64(mf[1])
    x_new = zero(x)
    for i=1:size(x_new,2)
        for j=1:div.(size(x_new,1),m)
            a = argmax(x[j*m-m+1:j*m, i])
            x_new[a+(j-1)*m,i] = g[j,i]
        end
    end
    return tuple(x_new, 1.0)
end

# m->mask size l->layers number
function extend_input(x::Matrix{Float32}, m::Int64, l::Int64)
    E = size(x,2)
    if E%(m^l)!=0
        E+=m^l-E%(m^l)
    end
    x_new = zeros(Float32,1,E)
    x_new[1:length(x)] = x
    return x_new
end

#CNN
conv(x::GraphNode, m::GraphNode) = BroadcastedOperator(conv,x,m)
forward(::BroadcastedOperator{typeof(conv)}, x, m) = return multi_convolution(x,m)
backward(::BroadcastedOperator{typeof(conv)}, x, m, g) = return dif_convolution(x,m,g)

max_pool(x::GraphNode, m::GraphNode) = BroadcastedOperator(max_pool,x,m)
forward(::BroadcastedOperator{typeof(max_pool)}, x, m) = return m_pool(x, m)
backward(::BroadcastedOperator{typeof(max_pool)}, x, m, g) = return dif_max_pool(x,m,g)

flatten(x::GraphNode) = BroadcastedOperator(flatten, x)
forward(::BroadcastedOperator{typeof(flatten)}, x) = begin
    new_x = zeros(Float32,size(x,1)*size(x,2),1)
    for i=1:length(new_x)
        new_x[i]=x[i]
    end
    return new_x
end
backward(::BroadcastedOperator{typeof(flatten)}, x, g) = begin
    dx = zeros(Float32,size(x)...)
    for i=1:length(dx)
        dx[i]=g[i]
    end
    return dx
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

update!(node::GraphNode, gradient) = let
    if isnothing(node.gradient)
        node.gradient = gradient 
    else
        node.gradient .+= gradient
    end
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