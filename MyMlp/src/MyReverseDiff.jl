module MyReverseDiff
export Variable

abstract type GraphNode end
abstract type Operator <: GraphNode end

# Definition of basic structures for computational graph
struct Constant{T<:Float64} <: GraphNode
    output :: T
end

mutable struct Variable{T<:Float64} <: GraphNode
    output :: T
    grad :: Union{Nothing, T}
    name :: String
    Variable(output::T; name="?") where T<:Float64 = new{T}(output, nothing, name)
end

mutable struct ScalarOperator{F} <: Operator
    inputs :: Any
    output :: Any
    grad :: Any
    name :: String
    ScalarOperator(fun, inputs...; name="?") = new{typeof(fun)}(inputs, nothing, nothing, name)
end

mutable struct BroadcastedOperator{F} <: Operator
    inputs :: Any
    output :: Any
    grad :: Any
    name :: String
    BroadcastedOperator(fun, inputs...; name="?") = new{typeof(fun)}(inputs, nothing, nothing, name)
end
# End of basic structures

end