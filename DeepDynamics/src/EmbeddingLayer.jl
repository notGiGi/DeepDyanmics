module EmbeddingLayer

using ..TensorEngine
using Random
using ..AbstractLayer

export Embedding, embedding_forward

mutable struct Embedding <: AbstractLayer.Layer
    vocab_size::Int
    embedding_dim::Int
    weights::TensorEngine.Tensor  # de tamaño (embedding_dim, vocab_size)
    trainable::Bool               # Permite congelar la capa si es necesario
end

function Embedding(vocab_size::Int, embedding_dim::Int)
    scale = 0.01
    w = scale .* Random.randn(embedding_dim, vocab_size)
    return Embedding(vocab_size, embedding_dim, TensorEngine.Tensor(w), true)
end

function forward(layer::Embedding, input_indices::Vector{Int})
    emb_matrix = layer.weights.data
    seq_len = length(input_indices)
    out = zeros(Float64, layer.embedding_dim, seq_len)
    for (i, idx) in enumerate(input_indices)
        out[:, i] = idx == 0 ? zeros(layer.embedding_dim) : emb_matrix[:, idx]
    end
    out_tensor = TensorEngine.Tensor(out)
    return out_tensor
end

# Soporte para Vector{Float64}
function forward(layer::Embedding, input_indices::Vector{Float64})
    indices_int = [Int(x) for x in input_indices]
    return forward(layer, indices_int)
end

const embedding_forward = forward

end  # End of module EmbeddingLayer
