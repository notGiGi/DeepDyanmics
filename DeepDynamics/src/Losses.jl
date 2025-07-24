module Losses
using ..TensorEngine
using CUDA

export binary_crossentropy, categorical_crossentropy

"""
    binary_crossentropy(y_pred, y_true)

Calcula la pérdida de entropía cruzada binaria.
"""
function binary_crossentropy(y_pred::TensorEngine.Tensor, y_true::TensorEngine.Tensor)
    # Detectar si estamos en GPU o CPU
    is_on_gpu = (y_pred.data isa CUDA.CuArray)
    
    # Asegurar que ambos tensores estén en el mismo dispositivo y tengan el mismo tipo
    y_pred_data = y_pred.data
    y_true_data = y_true.data
    
    # Obtener el tipo de datos (Float32/Float64)
    data_type = eltype(y_pred_data)
    
    if is_on_gpu && !(y_true_data isa CUDA.CuArray)
        y_true_data = CUDA.CuArray{data_type}(convert.(data_type, y_true_data))
    elseif !is_on_gpu && (y_true_data isa CUDA.CuArray)
        y_true_data = Array{data_type}(convert.(data_type, Array(y_true_data)))
    else
        # Asegurar que ambos tengan el mismo tipo de datos
        if eltype(y_true_data) != data_type
            if is_on_gpu
                y_true_data = CUDA.CuArray{data_type}(convert.(data_type, Array(y_true_data)))
            else
                y_true_data = convert.(data_type, y_true_data)
            end
        end
    end
    
    # Calculamos la pérdida con estabilidad numérica
    epsilon = convert(data_type, 1e-7)
    p = clamp(y_pred_data[1,1], epsilon, 1 - epsilon)
    loss_val = -(y_true_data[1,1] * log(p) + (1 - y_true_data[1,1]) * log(1 - p))
    
    # Crear tensor de resultado en el mismo dispositivo y tipo
    result_data = if is_on_gpu 
        CUDA.fill(loss_val, (1,1))
    else
        reshape([loss_val], (1,1))
    end
    
    result = TensorEngine.Tensor(result_data)
    
    # Solo asignar backward_fn si y_pred requiere gradientes
    if y_pred.requires_grad
        result.backward_fn = grad -> begin
            # Manejar tanto escalares como arrays
            grad_val = if grad isa AbstractArray
                length(grad) == 1 ? grad[1] : error("Gradiente debe ser escalar")
            else
                grad
            end
            
            # Asegurar que grad_val tiene el tipo correcto
            grad_val = convert(data_type, grad_val)
            
            # Calcular gradiente
            dLdp = -(y_true_data[1,1] / p - (1 - y_true_data[1,1]) / (1 - p))
            
            # Preparar el tensor de gradiente en el dispositivo y tipo correctos
            if is_on_gpu
                grad_tensor = TensorEngine.Tensor(CUDA.fill(grad_val * dLdp, size(y_pred_data)))
            else
                grad_tensor = TensorEngine.Tensor(fill(grad_val * dLdp, size(y_pred_data)))
            end
            
            # Propagar gradiente
            TensorEngine.backward(y_pred, grad_tensor)
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