# MyEmbedding.jl
module MyEmbedding

using ..MyReverseDiff
import ..MyReverseDiff: GraphNode, BroadcastedOperator, forward, backward

export embedding


embedding(embeddings_node::GraphNode, indices_node::GraphNode; name="embedding") = BroadcastedOperator(embedding, embeddings_node, indices_node, name=name)


function forward(::BroadcastedOperator{typeof(embedding)}, embeddings_val::Matrix{Float32}, indices_val::Matrix{Float32})
    embedding_dim, vocab_size = size(embeddings_val)
    sequence_length, batch_size = size(indices_val)

    result = zeros(Float32, embedding_dim, sequence_length, batch_size)

    actual_indices = round.(Int, indices_val)

    for b in 1:batch_size
        for s in 1:sequence_length
            idx = actual_indices[s, b]
            if idx >= 1 && idx <= vocab_size
                @views result[:, s, b] = embeddings_val[:, idx]
            else
                @views result[:, s, b] .= 0.0f0
            end
        end
    end

    return result
end


function backward(node::BroadcastedOperator{typeof(embedding)}, embeddings_val::Matrix{Float32}, indices_val::Matrix{Float32}, g::AbstractArray{Float32})

    embedding_node_g = zeros(Float32, size(embeddings_val))
    batch_size = size(node.output, 3)

    
    padding_index = size(embeddings_val, 2)
    
    for b in 1:batch_size
        for row in 1:size(indices_val, 1)
            idx = round(Int, indices_val[row, b])
            if idx >= 1 && idx <= size(embeddings_val, 2)
                embedding_node_g[:, idx] += g[:, row, b]
            end
        end
    end

    if padding_index >= 1 && padding_index <= size(embeddings_val, 2)
        embedding_node_g[:, padding_index] .= 0.0f0
    end
    
    return (embedding_node_g,)
end

end