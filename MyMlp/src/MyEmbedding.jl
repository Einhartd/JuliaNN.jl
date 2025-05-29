# MyEmbedding.jl
module MyEmbedding

using ..MyReverseDiff
import ..MyReverseDiff: GraphNode, BroadcastedOperator, forward, backward

export embedding


embedding(embeddings_node::GraphNode, indices_node::GraphNode; name="embedding") = BroadcastedOperator(embedding, embeddings_node, indices_node, name=name)


function forward(::BroadcastedOperator{typeof(embedding)}, embeddings_val::Matrix{Float32}, indices_val::Matrix{Float32})

    # embeddings_val: Macierz embeddingów (embedding_dim, vocab_size)
    # indices_val: Macierz indeksów słów (batch_size, sequence_length) - zakładamy taką konwencję

    embedding_dim, vocab_size = size(embeddings_val)
    sequence_length, batch_size = size(indices_val)
    
    # Wynik: 3D tensor (embedding_dim, sequence_length, batch_size)
    result = zeros(Float32, embedding_dim, sequence_length, batch_size) # Nadal alokuje tymczasowo, compute! kopiuje

    actual_indices = round.(Int, indices_val) # Indeksy muszą być całkowite

    # Wypełnianie tensora wyników
    for b in 1:batch_size
        for s in 1:sequence_length
            idx = actual_indices[s, b]
            # Sprawdzenie zakresu indeksów
            if idx >= 1 && idx <= vocab_size
                @views result[:, s, b] = embeddings_val[:, idx] # Użyj @views dla efektywności
            else
                # Domyślne zerowanie dla indeksów poza zakresem / paddingu
                @views result[:, s, b] .= 0.0f0
            end
        end
    end

    return result
end


function backward(node::BroadcastedOperator{typeof(embedding)}, embeddings_val::Matrix{Float32}, indices_val::Matrix{Float32}, g::AbstractArray{Float32})

    embeddings_node_ref = node.inputs[1] # Referencja do węzła z embeddingami
    # create a new matrix for gradients
    embedding_node_g = zeros(Float32, size(embeddings_val)) # Gradienty dla embeddingów
    batch_size = size(node.output, 3) # Rozmiar batcha

    # get index of padding (last column in embeddings_val)
    padding_index = size(embeddings_val, 2)
    
    for b in 1:batch_size
        for row in 1:size(indices_val, 1)
            idx = round(Int, indices_val[row, b]) # Indeks słowa
            if idx >= 1 && idx <= size(embeddings_val, 2)
                # Dodaj gradient do odpowiedniego indeksu
                embedding_node_g[:, idx] += g[:, row, b]
            end
        end
    end

    # Sprawdzenie, czy padding_index jest w zakresie
    if padding_index >= 1 && padding_index <= size(embeddings_val, 2)
        # wyzeruj gradient dla padding_index
        embedding_node_g[:, padding_index] .= 0.0f0
    end
    
    return (embedding_node_g,) # Zwracamy referencję do zaktualizowanego gradientu
end

end # Koniec modułu MyEmbedding