module Losses
using ..TensorEngine
using CUDA
using Statistics
using NNlib: σ
export binary_crossentropy, categorical_crossentropy, binary_crossentropy_with_logits

"""
    binary_crossentropy(y_pred, y_true)

Calcula la pérdida de entropía cruzada binaria con soporte CPU/GPU.
"""
function binary_crossentropy(
    y_pred::TensorEngine.Tensor,
    y_true::TensorEngine.Tensor
)
    # 1) Asegurar mismo dispositivo
    is_on_gpu   = (y_pred.data isa CUDA.CuArray)
    y_true_data = y_true.data
    if is_on_gpu && !(y_true_data isa CUDA.CuArray)
        y_true_data = CUDA.CuArray(y_true_data)
    elseif !is_on_gpu && (y_true_data isa CUDA.CuArray)
        y_true_data = Array(y_true_data)
    end

    # 2) Forward: BCE elemento a elemento
    ε = 1f-7
    p_clipped = clamp.(y_pred.data, ε, 1f0 - ε)
    elems = -(
        y_true_data .* log.(p_clipped) .+
        (1f0 .- y_true_data) .* log.(1f0 .- p_clipped)
    )
    # Promedio
    N = length(elems)
    loss_val = sum(elems) / N

    # 3) Crear tensor de salida
    result = TensorEngine.Tensor(reshape([loss_val], (1,1)); requires_grad=true)

    # 4) Backward: (p - y) / N
    if y_pred.requires_grad
        result.backward_fn = function(grad)
            gs = grad isa Number ? grad : grad[1]
            grad_input = (y_pred.data .- y_true_data) .* gs ./ N
            TensorEngine.backward(y_pred, TensorEngine.Tensor(grad_input))
        end
    end

    return result
end


# En tu módulo de pérdidas (p. ej. `Losses.jl`)

"""
    binary_crossentropy_with_logits(logits::Tensor, targets::Tensor)

Pérdida de entropía cruzada binaria en logits (capa lineal) con soporte CPU/GPU.
Forward:
  L = mean( max(z,0) - z⋅y + log(1 + exp(-|z|)) )
Backward implícita en el grafo:
  dL/dz = sigmoid(z) .- y
"""
function binary_crossentropy_with_logits(
    logits::TensorEngine.Tensor,
    y_true::TensorEngine.Tensor
)
    # ---- 1) Asegurar mismo dispositivo ----
    is_on_gpu = (logits.data isa CUDA.CuArray)
    t = y_true.data
    if is_on_gpu && !(t isa CUDA.CuArray)
        t = CUDA.CuArray(t)
    elseif !is_on_gpu && (t isa CUDA.CuArray)
        t = Array(t)
    end

    # ---- 2) Cómputo numéricamente estable ----
    z = logits.data
    # max(z,0)    → clamp.(z, 0, Inf)
    # log(1+e^{-abs(z)})
    absz = abs.(z)
    log_term = log.(1f0 .+ exp.(-absz))
    max_term = clamp.(z, 0f0, Inf)

    loss_elems = max_term .- z .* t .+ log_term
    loss_val = sum(loss_elems) / length(loss_elems)

    # ---- 3) Tensor resultado ----
    result = TensorEngine.Tensor(reshape([loss_val], (1,1));
                                 requires_grad=true)

    if logits.requires_grad
        result.backward_fn = function(grad)
            # grad puede ser escalar o Tensor 1×1
            gs = grad isa Number ? grad : grad[1]
            N  = length(z)
            σz = 1f0 ./ (1f0 .+ exp.(-z))
            # ∂L/∂z = (σ(z) - y) / N
            grad_input = (σz .- t) .* (gs / N)
            TensorEngine.backward(logits,
                TensorEngine.Tensor(grad_input))
        end
    end

    return result
end




"""
    categorical_crossentropy(y_pred, y_true)

Calcula la pérdida de entropía cruzada categórica para clasificación multiclase.
Asume que y_pred ya contiene probabilidades (después de softmax).
"""
function categorical_crossentropy(y_pred::TensorEngine.Tensor, y_true::TensorEngine.Tensor)::TensorEngine.Tensor
    # Detectar si estamos en GPU o CPU
    is_on_gpu = (y_pred.data isa CUDA.CuArray)
    
    # Obtener dimensiones
    pred_shape = size(y_pred.data)
    true_shape = size(y_true.data)
    
    # Comprobar que las dimensiones de batch coinciden
    if pred_shape[2] != true_shape[2]
        error("Dimensiones de batch no coinciden: y_pred $(pred_shape), y_true $(true_shape)")
    end
    
    # Asegurar que ambos tensores estén en el mismo dispositivo y tipo
    y_pred_data = y_pred.data
    y_true_data = y_true.data
    
    # Asegurar que ambos usen Float32 para consistencia
    y_pred_type = eltype(y_pred_data)
    y_true_type = eltype(y_true_data)
    
    # Convertir al mismo tipo (Float32)
    if y_pred_type != Float32
        if is_on_gpu
            y_pred_data = CUDA.convert.(Float32, y_pred_data)
        else
            y_pred_data = convert.(Float32, y_pred_data)
        end
    end
    
    if y_true_type != Float32
        if is_on_gpu
            y_true_data = CUDA.convert.(Float32, y_true_data)
        else
            y_true_data = convert.(Float32, y_true_data)
        end
    end
    
    # Asegurar que estén en el mismo dispositivo
    if is_on_gpu && !(y_true_data isa CUDA.CuArray)
        y_true_data = CUDA.CuArray{Float32}(y_true_data)
    elseif !is_on_gpu && (y_true_data isa CUDA.CuArray)
        y_true_data = Array{Float32}(y_true_data)
    end
    
    # Estabilidad numérica
    ε = 1.0f-7
    
    # Calcular pérdida
    if is_on_gpu
        # Versión GPU
        clipped_pred = CUDA.clamp.(y_pred_data, ε, 1.0f0 - ε)
        loss_per_sample = -CUDA.sum(y_true_data .* CUDA.log.(clipped_pred), dims=1)
        loss_val = CUDA.sum(loss_per_sample) / Float32(size(loss_per_sample, 2))
        result = TensorEngine.Tensor(CUDA.reshape([loss_val], (1, 1)))
    else
        # Versión CPU
        clipped_pred = clamp.(y_pred_data, ε, 1.0f0 - ε)
        loss_per_sample = -sum(y_true_data .* log.(clipped_pred), dims=1)
        loss_val = sum(loss_per_sample) / Float32(size(loss_per_sample, 2))
        result = TensorEngine.Tensor(reshape([loss_val], (1, 1)))
    end
    
    # Solo asignar backward_fn si y_pred requiere gradientes
    if y_pred.requires_grad
        result.backward_fn = grad -> begin
            # Asegurar que grad es Float32 y está en el dispositivo correcto
            grad_data = grad
            
            # Convertir tipo de datos
            if eltype(grad_data) != Float32
                if is_on_gpu
                    grad_data = CUDA.convert.(Float32, grad_data)
                else
                    grad_data = convert.(Float32, grad_data)
                end
            end
            
            # Asegurar mismo dispositivo
            if (grad_data isa CUDA.CuArray) != is_on_gpu
                if is_on_gpu
                    grad_data = CUDA.CuArray{Float32}(grad_data)
                else
                    grad_data = Array{Float32}(grad_data)
                end
            end
            
            # Calcular gradiente: (probs - targets) / batch_size
            batch_size = Float32(size(y_pred_data, 2))
            
            # Escalar gradiente
            scaled_grad = (y_pred_data .- y_true_data) .* (grad_data ./ batch_size)
            
            # Propagar gradiente
            TensorEngine.backward(y_pred, TensorEngine.Tensor(scaled_grad))
        end
    end
    
    return result
end

end  # module Losses