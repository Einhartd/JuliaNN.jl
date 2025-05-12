module MyReverseDiff
export Variable

abstract type GraphNode end
abstract type Operator <: GraphNode end

# Definition of basic structures for computational graph
struct Constant{T<:Float64} <: GraphNode
    output :: T
end

mutable struct Variable <: GraphNode
    output :: Matrix{Float64}
    grad :: Union{Nothing, Matrix{Float64}}
    name :: String
    Variable(output::Matrix{Float64}; name="?") = new(output, nothing, name)
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
    output :: Matrix{Float64}
    grad :: Any
    name :: String
    BroadcastedOperator(fun, inputs...; name="?") = new{typeof(fun)}(inputs, nothing, nothing, name)
end
# End of basic structures
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
    #   Sprawdza, czy node jest już w zbiorze visited. ∈
    if node ∈ visited
    # Do nothing, already visisted
    else
        #   Dodaje node do visited i do wektora order.
        push!(visited, node)   # Dodaje element na koniec zbioru visited
        push!(order, node)   # Dodaje element na koniec wektora order
    end
    #   Zwraca nic.
    return nothing
end

end