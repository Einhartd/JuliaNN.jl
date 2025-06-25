"""
MyMlp.jl

High-level neural network components including layers, optimizers, and model building.
Supports both MLPs and CNNs with various layer types and Adam optimization.
"""
module MyMlp

export xavier_normal, xavier_uniform,
       xavier_normal!, xavier_uniform!,
       Dense, Embedding, ConvolutionBlock, PoolingBlock, FlattenBlock, Chain, TransposeBlock,
       Adam, AdamState,
       setup_optimizer, step!,
       build_graph!,
       collect_model_parameters,
       show, summary, reset!

using ..MyReverseDiff
using ..MyEmbedding
using Distributions

# Weight initialization functions
function xavier_uniform(size::Tuple{Int, Int})
    limit = sqrt(6.0f0 / (size[1] + size[2]))
    return Float32.(rand(Uniform(-limit, limit), size))
end

function xavier_uniform(size::Tuple{Int, Int, Int})
    limit = sqrt(6.0f0 / (size[1] + size[2] + size[3]))
    return Float32.(rand(Uniform(-limit, limit), size))
end

function xavier_normal(size::Tuple{Int, Int})
    limit = sqrt(2.0f0 / (size[1] + size[2]))
    return Float32.(rand(Normal(0.0f0, limit), size))
end

function xavier_uniform!(w::Matrix{Float32})
    fan_out, fan_in = size(w)
    limit = sqrt(6.0f0 / (fan_in + fan_out))
    Float32.(rand!(Uniform(-limit, limit), w))
end

function xavier_normal!(w::Matrix{Float32})
    fan_out, fan_in = size(w)
    limit = sqrt(2.0f0 / (fan_in + fan_out))
    Float32.(rand!(Normal(0.0f0, limit), w))
end

abstract type Layer end

# Dense (fully-connected) layer: output = activation(W * input + b)
mutable struct Dense <: Layer
    W::Variable
    b::Variable
    activation
    name::String
end

function Dense(in_features::Int, out_features::Int, activation=identity; 
    weight_init = xavier_uniform,
    bias_init = (dims) -> zeros(Float32, dims),
    name="dense")

    W = Variable(weight_init((out_features, in_features)); name="$(name)_w")
    b = Variable(bias_init((out_features, 1)); name="$(name)_b")

    return Dense(W, b, activation, name)
end

function (d::Dense)(x::GraphNode)
    multiplication_code = *(d.W, x, name="$(d.name)_mul")
    linear_output = +(multiplication_code, d.b, name="$(d.name)_add")

    if d.activation == relu
        return relu(linear_output, name="$(d.name)_relu")
    elseif d.activation == σ
        return σ(linear_output, name="$(d.name)_sigmoid")
    else
        try
             return d.activation(linear_output, name="$(d.name)_$(string(nameof(d.activation)))")
        catch
             return d.activation(linear_output)
        end
    end
end

# Transpose layer for reshaping tensors
mutable struct TransposeBlock <:Layer
    name::String
end

function TransposeBlock(;name="transposition")
    return TransposeBlock(name)
end

function (t::TransposeBlock)(x::GraphNode)
    tr = MyReverseDiff.transpose(x)
    tr.name = t.name
    return tr
end

# Embedding layer for discrete token to dense vector mapping
mutable struct Embedding <: Layer
    W::Variable
    name::String
end

function Embedding(vocab_size::Int, embedding_dim::Int;
                   weight_init = (dims) -> xavier_uniform(dims),
                   name="embedding_layer")

    W_val = weight_init((embedding_dim, vocab_size))
    W = MyReverseDiff.Variable(W_val; name="$(name)_W")

    return Embedding(W, name)
end

function Embedding(initial_weights::Matrix{Float32};
                   name="embedding_layer")
    
    W = MyReverseDiff.Variable(initial_weights; name="$(name)_W")

    return Embedding(W, name)
end

function (e::Embedding)(x::MyReverseDiff.GraphNode)
    return MyEmbedding.embedding(e.W, x; name="$(e.name)_output")
end

# Sequential model container
mutable struct Chain
    layers::Vector{<:Layer}
end

Chain(layers...) = Chain([layers...])

function (c::Chain)(x::GraphNode)
    input = x
    for layer in c.layers
        input = layer(input)
    end
    return input
end

# Build complete computational graph for training
function build_graph!(model::Chain, loss_fn, input_node::GraphNode, label_node::GraphNode; loss_name="loss")

    model_output_node = model(input_node)
    loss_node = loss_fn(model_output_node, label_node; name=loss_name)

    if hasproperty(loss_node, :name)
        loss_node.name = loss_name
    end

    order = topological_sort(loss_node)

    return (loss_node, model_output_node, order)
end

# Adam optimizer
abstract type AbstractOptimizer end

struct Adam <: AbstractOptimizer
    α :: Float32
    β1 :: Float32
    β2 :: Float32
    ε :: Float32
end

Adam(;a=0.001f0) = Adam(a, 0.9f0, 0.999f0, 1e-8)

mutable struct AdamState
    hyperparams :: Adam
    m :: Dict{String, Array{Float32}}
    v :: Dict{String, Array{Float32}}
    t :: Int
    parameters :: Vector{Tuple{String, Variable}}
end

function setup_optimizer(optimizer_config::AbstractOptimizer, model::Chain)
    trainable_vars = collect_model_parameters(model)
    m = Dict{String, Array{Float32}}()
    v = Dict{String, Array{Float32}}()
    for (name, var) in trainable_vars
        m[name] = zeros(Float32, size(var.output))
        v[name] = zeros(Float32, size(var.output))
    end
    return AdamState(optimizer_config, m, v, 0, trainable_vars)
end

function collect_model_parameters(model::Chain)
    all_params = Vector{Tuple{String, Variable}}()
    for layer in model.layers
        append!(all_params, collect_model_parameters(layer))
    end
    return all_params
end

function collect_model_parameters(layer::Dense)
    return [(layer.W.name, layer.W), (layer.b.name, layer.b)]
end

function collect_model_parameters(layer::Embedding)
    return [(layer.W.name, layer.W)]
end

function collect_model_parameters(layer::ConvolutionBlock)
    return [(layer.masks.name, layer.masks),(layer.bias.name, layer.bias)]
end

# Adam optimization step
function step!(optimizer_state::AdamState)
    optimizer_state.t += 1

    config = optimizer_state.hyperparams

    for (name, var) in optimizer_state.parameters
        g = var.gradient

        optimizer_state.m[name] = config.β1 * optimizer_state.m[name] + (1.0f0 - config.β1) * g
        optimizer_state.v[name] = config.β2 * optimizer_state.v[name] + (1.0f0 - config.β2) * (g .* g)

        m_corrected = optimizer_state.m[name] ./ (1.0f0 - config.β1 ^ optimizer_state.t)
        v_corrected = optimizer_state.v[name] ./ (1.0f0 - config.β2 ^ optimizer_state.t)

        var.output .-= config.α .* m_corrected ./ (sqrt.(v_corrected) .+ config.ε)
    end
end

function reset!(optimizer_state::AdamState)
    optimizer_state.t = 0
    for (name, var) in optimizer_state.parameters
        optimizer_state.m[name] .= zeros(size(var.output))
        optimizer_state.v[name] .= zeros(size(var.output))
    end
end

# Convolution layer
mutable struct ConvolutionBlock <: Layer
    masks::Variable
    bias::Variable
    act_fun::Function
    name::String
end

function ConvolutionBlock(mask_x::Int, mask_y::Int,mask_z::Int;
    weight_init = xavier_uniform,
    act_fun=relu,
    name="convolution")

    masks = Variable(weight_init((mask_x, mask_y, mask_z)); name="$(name)_masks_w")
    bias = Variable(zeros(Float32,1,mask_z),name="$(name)_masks_b")

    return ConvolutionBlock(masks, bias, act_fun, name)
end

function (c::ConvolutionBlock)(x::GraphNode)
    cl = conv(x,c.masks)
    cl.name = "$(c.name)_conv"

    bl = cl + c.bias
    bl.name = "$(c.name)_bias"

    rl = c.act_fun(bl)
    rl.name = "$(c.name)_activation"

    return rl
end

# Max pooling layer
mutable struct PoolingBlock <: Layer
    name::String
    pool_fun::Function
    pool_size::Constant
end

function PoolingBlock(size::Int;name="pooling",pool_fun=max_pool)
    pool_size = Constant([Float32(size);;])
    return PoolingBlock(name,pool_fun,pool_size)
end

function (p::PoolingBlock)(x::GraphNode)
    pl = max_pool(x,p.pool_size)
    pl.name = "$(p.name)_pool"
    return pl
end

# Flatten layer for converting 3D to 2D
mutable struct FlattenBlock <: Layer
    name::String
end

function FlattenBlock(;name="flatten")
    return FlattenBlock(name)
end

function (f::FlattenBlock)(x::GraphNode)
    fl = flatten(x)
    fl.name = f.name
    return fl
end

# Parameter collection for layers without parameters
function collect_model_parameters(::FlattenBlock)
    return []
end

function collect_model_parameters(::PoolingBlock)
    return []
end

function collect_model_parameters(::TransposeBlock)
    return []
end

end # module MyMlp