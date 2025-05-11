module MyMlp

# Import ReverseDiff module
include("MyReverseDiff.jl")
export MyReverseDiff

export greet

greet() = print("Hello World!")
end # module MyMlp
