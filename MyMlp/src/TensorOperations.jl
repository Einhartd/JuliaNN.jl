"""
TensorOperations.jl

Moduł rozszerzający standardowe operacje Julia o mnożenie tensorów 3D.
Umożliwia mnożenie macierzowe tensorów 3D przez macierze 2D i wektory.

Przykład użycia:
```julia
using .TensorOperations

x = randn(Float32, 2, 2, 2)  # tensor 3D
y = [1.0f0; 2.0f0]           # wektor
z = x * y                    # mnożenie tensorowe
```
"""
module TensorOperations

function Base.:*(tensor::Array{T,3}, matrix::Array{T,2}) where T
    # Sprawdzenie zgodności wymiarów dla mnożenia macierzowego
    if size(tensor, 2) != size(matrix, 1)
        throw(DimensionMismatch("Dla mnożenia macierzowego tensor[:,:,k] * matrix: " *
                               "size(tensor,2)=$(size(tensor,2)) musi być równe " *
                               "size(matrix,1)=$(size(matrix,1))"))
    end
    
    # Wynik będzie miał wymiary (size(tensor,1), size(matrix,2), size(tensor,3))
    result = zeros(T, size(tensor, 1), size(matrix, 2), size(tensor, 3))
    
    # Mnożenie macierzowe każdej warstwy tensora przez macierz
    for k in 1:size(tensor, 3)
        result[:, :, k] = tensor[:, :, k] * matrix
    end
    
    return result
end


function Base.:*(tensor::Array{T,3}, vector::Vector{T}) where T
    # Sprawdzenie zgodności wymiarów
    if size(tensor, 2) != length(vector)
        throw(DimensionMismatch("Dla mnożenia macierzowego tensor[:,:,k] * vector: " *
                               "size(tensor,2)=$(size(tensor,2)) musi być równe " *
                               "length(vector)=$(length(vector))"))
    end
    
    # Wynik będzie miał wymiary (size(tensor,1), 1, size(tensor,3))
    result = zeros(T, size(tensor, 1), 1, size(tensor, 3))
    
    # Mnożenie macierzowe każdej warstwy tensora przez wektor
    for k in 1:size(tensor, 3)
        result[:, 1, k] = tensor[:, :, k] * vector
    end
    
    return result
end


end # module TensorOperations