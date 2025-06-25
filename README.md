# JuliaNN.jl

A lightweight neural network library built from scratch in Julia, featuring automatic differentiation and support for both MLP and CNN architectures.

## Work in Progress

**This project is currently under development and is not production-ready.** Many features are experimental and may contain bugs. Use at your own risk for educational purposes.

## Features

- **Custom Reverse-Mode Automatic Differentiation** - Built-in backpropagation engine
- **Multiple Layer Types** - Dense, Embedding, Convolution, Pooling, and more
- **Flexible Architecture** - Support for both MLPs and CNNs
- **Adam Optimizer** - Adaptive learning rate optimization
- **Loss Functions** - Binary cross-entropy with more to come
- **Computational Graph** - Explicit graph construction and topological sorting

## Quick Start

```julia
include("MyReverseDiff.jl")
include("MyEmbedding.jl")
include("MyMlp.jl")

using .MyReverseDiff
using .MyMlp

# Create a CNN model (like in the IMDB example)
batch_size = 64
input_size = 50  # sequence length

model = Chain(
    Embedding(embeddings, name="embedding"),
    TransposeBlock(),
    ConvolutionBlock(3, 50, 8, name="layer1"),
    PoolingBlock(8),
    FlattenBlock(name="flatten"),
    Dense(input_size-2, 1, σ, name="output")
)

# Set up training nodes
x_input_node = Constant(zeros(Float32, input_size, batch_size))
y_label_node = Constant(zeros(Float32, 1, batch_size))

# Build computational graph
loss_node, model_output_node, order = build_graph!(model, binarycrossentropy, x_input_node, y_label_node)

# Setup optimizer
optimizer_state = setup_optimizer(Adam(a=0.001f0), model)

# Training loop
for epoch in 1:epochs
    for batch_data, batch_labels in data_loader
        # Update input nodes with batch data
        x_input_node.output .= batch_data
        y_label_node.output .= batch_labels
        
        # Forward and backward pass
        forward!(order)
        backward!(order)
        step!(optimizer_state)
    end
end
```

## CNN Example

```julia
# Build a CNN for text classification (IMDB sentiment analysis)
model = Chain(
    Embedding(embeddings, name="embedding"),           # Word embeddings
    TransposeBlock(),                                  # Reshape for convolution
    ConvolutionBlock(3, 50, 8, name="conv1"),         # 1D convolution
    PoolingBlock(8),                                   # Max pooling
    FlattenBlock(name="flatten"),                      # Flatten for dense layer
    Dense(input_size-2, 1, σ, name="classifier")      # Binary classification
)

# Create input constants (not variables)
x_input_node = Constant(zeros(Float32, sequence_length, batch_size))
y_label_node = Constant(zeros(Float32, 1, batch_size))

# Build graph and train
loss_node, output_node, order = build_graph!(model, binarycrossentropy, x_input_node, y_label_node)
```

## Architecture

The library consists of three main modules:

- **`MyReverseDiff.jl`** - Core automatic differentiation engine
- **`MyEmbedding.jl`** - Embedding layer implementation  
- **`MyMlp.jl`** - High-level neural network components

### Supported Operations

- Matrix multiplication and addition
- Activation functions (ReLU, Sigmoid)
- Convolution and max pooling
- Embedding lookups
- Transposition and flattening

## Installation

```julia
# Clone the repository
git clone https://github.com/yourusername/JuliaNN.jl.git

# Include in your project
include("path/to/JuliaNN.jl/MyReverseDiff.jl")
include("path/to/JuliaNN.jl/MyMlp.jl")

using .MyReverseDiff, .MyMlp
```

## Examples

Check out the `examples/` directory for complete training examples:

- **IMDB Sentiment Analysis** - Text classification with embeddings and CNNs
- **Basic MLP** - Simple feedforward networks
- **Custom Layers** - How to extend the library

## Current Limitations

- Limited to Float32 precision
- No GPU support
- Basic optimizer selection (Adam only)
- Experimental stability
- Limited documentation

## Authors

- **Einhartd**
- **SzymonRogozinski**

## Acknowledgments

- Built as a learning exercise in automatic differentiation
- Inspired by modern deep learning frameworks
- Thanks to the Julia community for excellent documentation

---

*Note: This library is for educational purposes. For production use, consider established frameworks like Flux.jl or MLJ.jl.*
