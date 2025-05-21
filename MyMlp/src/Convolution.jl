using ..MyMlp
using ..MyReverseDiff

mutable struct ConvolutionBlock <: Layer
    masks::Variable
    pool_fun::Function
    pool_size::Constant
    act_fun::Function
    name::String
end

function ConvolutionBlock(mask_count::Int, mask_size::Int, size::Int;
    weight_init = xavier_uniform,
    pool_fun=max_pool,
    act_fun=relu,
    name="convolution")

    masks = Variable(weight_init((mask_count, mask_size)); name="$(name)_masks")
    pool_size = Constant([Float32(size);;])

    return ConvolutionBlock(masks, pool_fun, pool_size, act_fun, name)
end

function (c::ConvolutionBlock)(x::GraphNode)
    cl = conv(x,c.masks)
    cl.name = "$(c.name)_conv"

    pl = c.pool_fun(cl,c.pool_size)
    pl.name = "$(c.name)_pool"

    al = c.act_fun(pl)
    al.name = "$(c.name)_active"

    return al
end

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

function collect_model_parameters(layer::FlattenBlock)
    return []
end

function collect_model_parameters(layer::ConvolutionBlock)
    return [(layer.masks.name, layer.masks)]
end

# Convolution
 
function convolution(x::Vector{Float64}, m::Vector{Float64})
    k = length(m)
    l = length(x)
    x_new = zeros(l)
    ll = l-k+1
    for i=1:ll
        x_new[i] = sum(x[i:i+k-1].*m)
    end
    for i=(ll+1):l
        x_new[i] = sum(x[i:l].*m[1:l-i+1])
    end
    return x_new
end

function multi_convolution(x::Matrix{Float64},m::Matrix{Float64})
    l = size(m,1)
    x_new = zeros(l*size(x,1),size(x,2))
    for j=1:size(x,1)
        for i=1:l
            x_new[(j-1)*l+i,:] = convolution(x[j,:],m[i,:])
        end
    end
    return x_new
end

function dif_convolution(x::Matrix{Float64}, m::Matrix{Float64}, g::Matrix{Float64})
    lx = size(x,2)
    dx = zeros(size(x,1),lx)

    size_diff = div(size(g,1),size(x,1))

    for i=1:size(g,1)
        dx[1+div(i-1,size_diff),:] .+= convolution(g[i,:],reverse(m[i%size_diff+1,:]))
    end

    tmp = multi_convolution(x,g)
    dm = zeros(size(m)...)
    for i=1:size(dm,1)
        t = tmp[i:size(dm,1):end, :]
        tt = sum(t,dims=1)
        dm[i,:] = reverse(tt[:,(lx-size(m,2)+1):lx], dims=2)
    end
    
    return tuple(dx,dm)
end

# Max Pool
function m_pool(x::Matrix{Float64},m::Int64)
    x_new = zeros(size(x,1),div(size(x,2),m))
    for i=1:size(x_new,1)
        for j=1:size(x_new,2)
            a = argmax(x[i,j*m-m+1:j*m])
            x_new[i,j] = x[i,j*m-m+a]
        end
    end
    return x_new
end

function dif_max_pool(x::Matrix{Float64},m::Int64, g::Matrix{Float64})
    x_new = zeros(size(x)...)
    for i=1:size(x_new,1)
        for j=1:div.(size(x_new)[2],m)
            a = argmax(x[i,j*m-m+1:j*m])
            x_new[i,a+(j-1)*m] = g[i,j]
        end
    end
    return tuple(x_new, 1.0)
end

# m->mask size l->layers number
function extend_input(x::Matrix{Float64}, m::Int64, l::Int64)
    E = size(x,2)
    if E%(m^l)!=0
        E+=m^l-E%(m^l)
    end
    x_new = zeros(1,E)
    x_new[1:length(x)] = x
    return x_new
end

#CNN
conv(x::GraphNode, m::GraphNode) = BroadcastedOperator(conv,x,m)
forward(::BroadcastedOperator{typeof(conv)}, x, m) = return multi_convolution(x,m)
backward(node::BroadcastedOperator{typeof(conv)}, x, m, g) = return dif_convolution(x,m,g)

max_pool(x::GraphNode, m::GraphNode) = BroadcastedOperator(max_pool,x,m)
forward(::BroadcastedOperator{typeof(m_pool)}, x, m) = return m_pool(x, m)
backward(node::BroadcastedOperator{typeof(m_pool)}, x, m, g) = return dif_max_pool(x,m,g)

flatten(x::GraphNode) = BroadcastedOperator(flatten, x)
forward(::BroadcastedOperator{typeof(flatten)}, x) = begin
    new_x = zeros(size(x,1)*size(x,2))
    for i=1:length(new_x)
        new_x[i]=x[i]
    end
    return new_x
end
backward(::BroadcastedOperator{typeof(flatten)}, x, g) = begin
    dx = zeros(size(x)...)
    for i=1:length(dx)
        dx[i]=g[i]
    end
    return dx
end
