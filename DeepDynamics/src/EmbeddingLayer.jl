module EmbeddingLayer

using ..TensorEngine
using Random
using ..AbstractLayer

export Embedding, embedding_forward, forward

mutable struct Embedding <: AbstractLayer.Layer
    vocab_size::Int
    embedding_dim::Int
    weights::TensorEngine.Tensor  # de tamaño (embedding_dim, vocab_size)
    trainable::Bool               # Permite congelar la capa si es necesario
end

"""
    Embedding(vocab_size::Int, embedding_dim::Int; trainable=true)

Crea una capa de embedding con inicialización Xavier mejorada.
"""
function Embedding(vocab_size::Int, embedding_dim::Int; trainable=true)
    # Inicialización Xavier/Glorot para embeddings
    scale = sqrt(2.0 / (vocab_size + embedding_dim))
    w = Float32.(scale .* Random.randn(embedding_dim, vocab_size))
    weights = TensorEngine.Tensor(w; requires_grad=trainable)
    return Embedding(vocab_size, embedding_dim, weights, trainable)
end

"""
    forward(layer::Embedding, input_indices::Vector{Int})

Forward pass con soporte para backward diferenciable.
"""
function forward(layer::Embedding, input_indices::Vector{Int})
    # NUEVO: Validar que no esté vacío
    if isempty(input_indices)
        throw(BoundsError("Cannot embed empty sequence"))
    end
    
    # Validar índices (código existente)
    for idx in input_indices
        if idx < 0 || idx > layer.vocab_size
            error("Índice $idx fuera de rango [0, $(layer.vocab_size)]")
        end
    end
    

    emb_matrix = layer.weights.data
    seq_len = length(input_indices)
        
    # Pre-allocar matriz de salida
    out_data = zeros(Float32, layer.embedding_dim, seq_len)
    
    # Extraer embeddings para cada índice
    for (i, idx) in enumerate(input_indices)
        if idx == 0  # Padding token
            out_data[:, i] .= 0.0f0
        else
            out_data[:, i] = emb_matrix[:, idx]
        end
    end
    
    # Crear tensor de salida
    out_tensor = TensorEngine.Tensor(out_data; requires_grad=layer.trainable)
    
    # Definir función backward si los pesos son entrenables
    if layer.trainable && layer.weights.requires_grad
        out_tensor.backward_fn = grad -> begin
            # grad es de tamaño (embedding_dim, seq_len)
            grad_data = grad isa TensorEngine.Tensor ? grad.data : grad
            
            # Acumular gradientes para cada índice usado
            grad_weights = zeros(Float32, size(layer.weights.data))
            
            for (i, idx) in enumerate(input_indices)
                if idx > 0  # Ignorar padding
                    # Acumular gradiente para este embedding
                    grad_weights[:, idx] .+= grad_data[:, i]
                end
            end
            
            # Propagar gradiente a los pesos
            TensorEngine.backward(layer.weights, TensorEngine.Tensor(grad_weights; requires_grad=false))
        end
    end
    
    return out_tensor
end

"""
    forward(layer::Embedding, input_indices::Vector{Float64})

Soporte para índices Float64 (conversión automática a Int).
"""
function forward(layer::Embedding, input_indices::Vector{Float64})
    indices_int = [round(Int, x) for x in input_indices]
    return forward(layer, indices_int)
end

"""
    forward(layer::Embedding, input_indices::Vector{Float32})

Soporte para índices Float32 (conversión automática a Int).
"""
function forward(layer::Embedding, input_indices::Vector{Float32})
    indices_int = [round(Int, x) for x in input_indices]
    return forward(layer, indices_int)
end

"""
    forward(layer::Embedding, input_tensor::TensorEngine.Tensor)

Forward pass cuando la entrada es un Tensor de índices.
"""
function forward(layer::Embedding, input_tensor::TensorEngine.Tensor)
    # Convertir tensor a vector de índices
    indices = vec(round.(Int, input_tensor.data))
    
    # Forward normal
    output = forward(layer, indices)
    
    # Si el tensor de entrada requiere gradientes, necesitamos propagarlos
    # (aunque en la práctica, los índices no suelen requerir gradientes)
    if input_tensor.requires_grad
        old_backward = output.backward_fn
        output.backward_fn = grad -> begin
            # Ejecutar backward normal para los pesos
            if old_backward !== nothing
                old_backward(grad)
            end
            # Para los índices, el gradiente sería sparse, pero generalmente no se usa
            # Podríamos implementar un SparseGradient si fuera necesario
        end
    end
    
    return output
end

# Hacer la capa callable
(layer::Embedding)(input) = forward(layer, input)

# Alias para compatibilidad
const embedding_forward = forward

# Funciones auxiliares

"""
    freeze!(layer::Embedding)

Congela los pesos del embedding para que no se actualicen durante el entrenamiento.
"""
function freeze!(layer::Embedding)
    layer.trainable = false
    layer.weights.requires_grad = false
end

"""
    unfreeze!(layer::Embedding)

Descongela los pesos del embedding para permitir actualizaciones.
"""
function unfreeze!(layer::Embedding)
    layer.trainable = true
    layer.weights.requires_grad = true
end

"""
    load_pretrained!(layer::Embedding, pretrained_weights::Matrix)

Carga pesos preentrenados en la capa de embedding.
"""
function load_pretrained!(layer::Embedding, pretrained_weights::Matrix)
    @assert size(pretrained_weights) == size(layer.weights.data) "Dimensiones incompatibles"
    layer.weights.data .= Float32.(pretrained_weights)
end

"""
    get_embedding(layer::Embedding, index::Int)

Obtiene el vector de embedding para un índice específico.
"""
function get_embedding(layer::Embedding, index::Int)
    if index < 1 || index > layer.vocab_size
        error("Índice $index fuera de rango [1, $(layer.vocab_size)]")
    end
    return layer.weights.data[:, index]
end

"""
    set_embedding!(layer::Embedding, index::Int, embedding::Vector)

Establece el vector de embedding para un índice específico.
"""
function set_embedding!(layer::Embedding, index::Int, embedding::Vector)
    if index < 1 || index > layer.vocab_size
        error("Índice $index fuera de rango [1, $(layer.vocab_size)]")
    end
    @assert length(embedding) == layer.embedding_dim "Dimensión incorrecta del embedding"
    layer.weights.data[:, index] .= Float32.(embedding)
end

# Exportar funciones adicionales
export freeze!, unfreeze!, load_pretrained!, get_embedding, set_embedding!

end  # module EmbeddingLayer