module EmbeddingLayer

using ..TensorEngine
using ..AbstractLayer
using Random
using CUDA

export Embedding, embedding_forward, forward
export freeze!, unfreeze!, load_pretrained!, get_embedding, set_embedding!

# -----------------------------
# Utils internos
# -----------------------------
@inline to_cpu(x) = x isa CUDA.CuArray ? Array(x) : x
@inline maybe_to_gpu(x, ref) = (ref isa CUDA.CuArray) ? CUDA.CuArray(x) : x

# -----------------------------
# Definición de la capa
# -----------------------------
mutable struct Embedding <: AbstractLayer.Layer
    vocab_size::Int
    embedding_dim::Int
    weights::TensorEngine.Tensor   # (E, V)
    trainable::Bool
end

function Embedding(vocab_size::Int, embedding_dim::Int; trainable::Bool=true)
    scale = sqrt(2.0 / (vocab_size + embedding_dim))
    w = Float32.(scale .* Random.randn(embedding_dim, vocab_size)) # (E, V)
    weights = TensorEngine.Tensor(w; requires_grad=trainable)
    return Embedding(vocab_size, embedding_dim, weights, trainable)
end

# -----------------------------
# Forward para Vector{Int}
# Salida: (T, E, 1)
# -----------------------------
function forward(layer::Embedding, input_indices::Vector{Int})
    T = length(input_indices)
    E = layer.embedding_dim
    V = layer.vocab_size
    T == 0 && throw(BoundsError("Cannot embed empty sequence"))

    if any(i -> i < 0 || i > V, input_indices)
        error("Índice fuera de rango [0, $V]")
    end

    # Mapear 0→1 para gather; luego enmascaramos a 0
    idx_fix = Int[i == 0 ? 1 : i for i in input_indices]

    # Pesos en CPU para gather (evita indexado escalar en GPU)
    Wgpu = layer.weights.data                 # (E, V) CuArray o Array
    Wcpu = to_cpu(Wgpu)                       # (E, V) Array
    emb  = @views Wcpu[:, idx_fix]            # (E, T)

    # Zeros para padding (índice 0)
    if any(==(0), input_indices)
        mask = (input_indices .== 0)
        @inbounds for t in 1:T
            if mask[t]
                @views emb[:, t] .= 0f0
            end
        end
    end

    # Layout que espera RNN (batch_first=false): (T, E, B) con B=1
    out_data = permutedims(reshape(emb, (E, T, 1)), (2, 3, 1))  # (T, E, 1)
    out_data = maybe_to_gpu(out_data, Wgpu)

    out = TensorEngine.Tensor(out_data; requires_grad=layer.weights.requires_grad)

    if layer.weights.requires_grad
        out.backward_fn = grad -> begin
            # grad llega como (T, E, 1)
            G = grad isa TensorEngine.Tensor ? grad.data : grad
            G = reshape(G, (T, E))                    # (T, E)
            Gcpu = to_cpu(G)                          # (T, E) CPU
            Gcpu = permutedims(Gcpu, (2, 1))          # (E, T)

            dWcpu = zeros(Float32, size(Wcpu))        # (E, V)
            @inbounds for t in 1:T
                i = input_indices[t]
                if i != 0
                    @views dWcpu[:, i] .+= Gcpu[:, t]
                end
            end

            dW = maybe_to_gpu(dWcpu, Wgpu)
            TensorEngine.backward(layer.weights, TensorEngine.Tensor(dW; requires_grad=false))
        end
    end

    return out
end

# -----------------------------
# Forward para Vector{Float32/64}
# -----------------------------
function forward(layer::Embedding, input_indices::Vector{Float64})
    return forward(layer, [round(Int, x) for x in input_indices])
end
function forward(layer::Embedding, input_indices::Vector{Float32})
    return forward(layer, [round(Int, x) for x in input_indices])
end

# -----------------------------
# Forward para Tensor de índices
# Admite 1D (T) o 2D (T, N)
# Salida: (T, E, N)
# -----------------------------
function forward(layer::Embedding, input_tensor::TensorEngine.Tensor)
    A = input_tensor.data
    nd = ndims(A)
    @assert nd == 1 || nd == 2 "Embedding espera 1D (T) o 2D (T,N) de índices"

    # Reusar ruta de vector si es 1D
    if nd == 1
        return forward(layer, round.(Int, to_cpu(A)))
    end

    T, N = size(A)
    E    = layer.embedding_dim
    V    = layer.vocab_size

    # Índices en CPU (evitar indexado escalar en GPU)
    idx      = round.(Int, to_cpu(A))        # (T, N)
    idx_flat = reshape(idx, T*N)             # (T*N)
    if any(i -> i < 0 || i > V, idx_flat)
        error("Índice fuera de rango [0, $V]")
    end
    idx_fix  = Int[i == 0 ? 1 : i for i in idx_flat]

    Wgpu = layer.weights.data                # (E, V)
    Wcpu = to_cpu(Wgpu)                      # (E, V)

    # Gather en CPU: (E, T*N)
    emb = @views Wcpu[:, idx_fix]

    # Zeros para padding
    if any(==(0), idx_flat)
        @inbounds for p in 1:(T*N)
            if idx_flat[p] == 0
                @views emb[:, p] .= 0f0
            end
        end
    end

    # Layout que espera RNN (batch_first=false): (T, E, N)
    out_data = permutedims(reshape(emb, (E, T, N)), (2, 3, 1))  # (T, E, N)
    out_data = maybe_to_gpu(out_data, Wgpu)

    out = TensorEngine.Tensor(out_data; requires_grad=layer.weights.requires_grad)

    if layer.weights.requires_grad
        out.backward_fn = grad -> begin
            # grad llega como (T, E, N)
            G  = grad isa TensorEngine.Tensor ? grad.data : grad
            Gp = permutedims(G, (2, 1, 3))                # (E, T, N)
            Gp = reshape(Gp, (E, T*N))                    # (E, T*N)

            Gp_cpu = to_cpu(Gp)
            dWcpu  = zeros(Float32, size(Wcpu))           # (E, V)

            @inbounds for p in 1:(T*N)
                i = idx_flat[p]
                if i != 0
                    @views dWcpu[:, i] .+= Gp_cpu[:, p]
                end
            end

            dW = maybe_to_gpu(dWcpu, Wgpu)
            TensorEngine.backward(layer.weights, TensorEngine.Tensor(dW; requires_grad=false))
        end
    end

    return out
end

# -----------------------------
# Azúcar sintáctico
# -----------------------------
(layer::Embedding)(input) = forward(layer, input)
const embedding_forward = forward

# -----------------------------
# Utilidades
# -----------------------------
function freeze!(layer::Embedding)
    layer.trainable = false
    layer.weights.requires_grad = false
end

function unfreeze!(layer::Embedding)
    layer.trainable = true
    layer.weights.requires_grad = true
end

function load_pretrained!(layer::Embedding, pretrained_weights::Matrix)
    @assert size(pretrained_weights) == size(layer.weights.data) "Dimensiones incompatibles"
    layer.weights.data .= Float32.(pretrained_weights)
end

function get_embedding(layer::Embedding, index::Int)
    if index < 1 || index > layer.vocab_size
        error("Índice $index fuera de rango [1, $(layer.vocab_size)]")
    end
    return layer.weights.data[:, index]
end

function set_embedding!(layer::Embedding, index::Int, embedding::Vector)
    if index < 1 || index > layer.vocab_size
        error("Índice $index fuera de rango [1, $(layer.vocab_size)]")
    end
    @assert length(embedding) == layer.embedding_dim "Dimensión incorrecta del embedding"
    layer.weights.data[:, index] .= Float32.(embedding)
end

end # module
