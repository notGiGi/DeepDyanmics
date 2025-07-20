module Layers

using NNlib, CUDA, ..TensorEngine, ..AbstractLayer, ..ConvKernelLayers, Statistics
export BatchNorm, Flatten, LayerActivation, set_training!, reset_running_stats!, GlobalAvgPool, ResidualBlock, create_residual_block, CustomFlatten, DropoutLayer
# Añadir en Layers.jl justo antes del end del módulo

# ==================================================================
# BatchNorm Optimizado (Fusión GPU)
# ==================================================================
"""
Versión corregida de BatchNormalization que usa Float32 explícitamente
"""
# BatchNorm Mejorado para DeepDynamics - Fase 4
# Esta implementación corrige todos los problemas identificados

"""
BatchNormalization mejorada con soporte completo GPU/CPU y cálculo correcto de estadísticas
"""
mutable struct BatchNorm <: AbstractLayer.Layer
    gamma::TensorEngine.Tensor
    beta::TensorEngine.Tensor
    running_mean::Array{Float32}  # Siempre en CPU
    running_var::Array{Float32}   # Siempre en CPU
    momentum::Float32
    epsilon::Float32
    training::Bool
    # Nuevos campos para mejor tracking
    num_batches_tracked::Int
end

function BatchNorm(channels::Int; momentum=Float32(0.1), epsilon=Float32(1e-5), training=true)
    # CRÍTICO: Asegurar que los parámetros sean entrenables
    gamma = TensorEngine.Tensor(ones(Float32, channels); requires_grad=true)
    beta = TensorEngine.Tensor(zeros(Float32, channels); requires_grad=true)
    
    # Running statistics siempre en CPU
    running_mean = zeros(Float32, channels)
    running_var = ones(Float32, channels)
    
    return BatchNorm(gamma, beta, running_mean, running_var, momentum, epsilon, training, 0)
end

function forward(layer::BatchNorm, input::TensorEngine.Tensor)
    x = input.data
    dims = size(x)
    ndims_x = ndims(x)
    
    # Detectar dispositivo
    is_on_gpu = x isa CUDA.CuArray
    
    # Asegurar Float32
    if eltype(x) != Float32
        x = is_on_gpu ? CUDA.convert.(Float32, x) : convert.(Float32, x)
    end
    
    if ndims_x == 4
        # Formato NCHW: (batch, channel, height, width)
        batch_size, num_channels, height, width = dims
        
        # Verificar dimensiones
        @assert num_channels == length(layer.gamma.data) "Número de canales no coincide"
        
        if layer.training
            # Calcular estadísticas del batch
            batch_mean = zeros(Float32, num_channels)
            batch_var = zeros(Float32, num_channels)
            
            # Si está en GPU, traer a CPU para cálculo de estadísticas
            x_cpu = is_on_gpu ? Array(x) : x
            
            # Calcular por canal
            for c in 1:num_channels
                channel_data = @view x_cpu[:, c, :, :]
                batch_mean[c] = mean(channel_data)
                batch_var[c] = var(channel_data, corrected=false)
            end
            
            # Actualizar running statistics
            layer.running_mean .= (1.0f0 - layer.momentum) .* layer.running_mean .+ layer.momentum .* batch_mean
            layer.running_var .= (1.0f0 - layer.momentum) .* layer.running_var .+ layer.momentum .* batch_var
            
            layer.num_batches_tracked += 1
            
            mean_use = batch_mean
            var_use = batch_var
        else
            # Modo eval: usar running statistics
            mean_use = layer.running_mean
            var_use = layer.running_var
        end
        
        # Preparar estadísticas para GPU si es necesario
        if is_on_gpu
            mean_use = CUDA.CuArray(mean_use)
            var_use = CUDA.CuArray(var_use)
        end
        
        # Normalizar
        mean_reshaped = reshape(mean_use, (1, num_channels, 1, 1))
        var_reshaped = reshape(var_use, (1, num_channels, 1, 1))
        std_reshaped = sqrt.(var_reshaped .+ layer.epsilon)
        
        x_normalized = (x .- mean_reshaped) ./ std_reshaped
        
        # Aplicar parámetros gamma y beta
        gamma_data = layer.gamma.data
        beta_data = layer.beta.data
        
        # Asegurar que gamma y beta estén en el dispositivo correcto
        if is_on_gpu
            if !(gamma_data isa CUDA.CuArray)
                gamma_data = CUDA.CuArray(gamma_data)
            end
            if !(beta_data isa CUDA.CuArray)
                beta_data = CUDA.CuArray(beta_data)
            end
        else
            if gamma_data isa CUDA.CuArray
                gamma_data = Array(gamma_data)
            end
            if beta_data isa CUDA.CuArray
                beta_data = Array(beta_data)
            end
        end
        
        gamma_reshaped = reshape(gamma_data, (1, num_channels, 1, 1))
        beta_reshaped = reshape(beta_data, (1, num_channels, 1, 1))
        
        output = gamma_reshaped .* x_normalized .+ beta_reshaped
        
    elseif ndims_x == 2
        # Formato (features, batch) para capas Dense
        num_features, batch_size = dims
        
        @assert num_features == length(layer.gamma.data) "Número de features no coincide"
        
        if layer.training
            # Calcular estadísticas sobre la dimensión del batch
            x_cpu = is_on_gpu ? Array(x) : x
            
            batch_mean = mean(x_cpu, dims=2)[:, 1]
            batch_var = var(x_cpu, dims=2, corrected=false)[:, 1]
            
            # Actualizar running stats
            layer.running_mean .= (1.0f0 - layer.momentum) .* layer.running_mean .+ layer.momentum .* batch_mean
            layer.running_var .= (1.0f0 - layer.momentum) .* layer.running_var .+ layer.momentum .* batch_var
            
            layer.num_batches_tracked += 1
            
            mean_use = batch_mean
            var_use = batch_var
        else
            mean_use = layer.running_mean
            var_use = layer.running_var
        end
        
        # Preparar para GPU si es necesario
        if is_on_gpu
            mean_use = CUDA.CuArray(mean_use)
            var_use = CUDA.CuArray(var_use)
        end
        
        # Normalizar
        mean_reshaped = reshape(mean_use, (num_features, 1))
        var_reshaped = reshape(var_use, (num_features, 1))
        std_reshaped = sqrt.(var_reshaped .+ layer.epsilon)
        
        x_normalized = (x .- mean_reshaped) ./ std_reshaped
        
        # Aplicar gamma y beta
        gamma_data = layer.gamma.data
        beta_data = layer.beta.data
        
        if is_on_gpu && !(gamma_data isa CUDA.CuArray)
            gamma_data = CUDA.CuArray(gamma_data)
            beta_data = CUDA.CuArray(beta_data)
        elseif !is_on_gpu && gamma_data isa CUDA.CuArray
            gamma_data = Array(gamma_data)
            beta_data = Array(beta_data)
        end
        
        gamma_reshaped = reshape(gamma_data, (num_features, 1))
        beta_reshaped = reshape(beta_data, (num_features, 1))
        
        output = gamma_reshaped .* x_normalized .+ beta_reshaped
        
    else
        error("BatchNorm: formato no soportado con $ndims_x dimensiones")
    end
    
    # CRÍTICO: Determinar si necesitamos gradientes
    # Necesitamos gradientes si:
    # 1. El input requiere gradientes, O
    # 2. Los parámetros (gamma/beta) requieren gradientes
    needs_grad = input.requires_grad || layer.gamma.requires_grad || layer.beta.requires_grad
    
    # Crear tensor de salida con requires_grad correcto
    out = TensorEngine.Tensor(output; requires_grad=needs_grad)
    
    # CRÍTICO: Definir backward incluso en modo eval si hay gradientes
    if needs_grad
        # Guardar valores necesarios para backward
        saved_mean = copy(mean_use)
        saved_var = copy(var_use)
        saved_normalized = copy(x_normalized)
        saved_std = sqrt.(saved_var .+ layer.epsilon)
        saved_x = copy(x)  # Guardar x original para gradientes de mean/var
        
        out.backward_fn = grad -> begin
            grad_data = grad isa TensorEngine.Tensor ? grad.data : grad
            
            # Asegurar mismo dispositivo
            if (grad_data isa CUDA.CuArray) != is_on_gpu
                if is_on_gpu
                    grad_data = CUDA.CuArray(grad_data)
                else
                    grad_data = Array(grad_data)
                end
            end
            
            if ndims_x == 4
                # Backward para 4D mejorado
                backward_batchnorm_4d_improved!(
                    layer, input, grad_data, 
                    saved_normalized, saved_mean, saved_std, saved_x,
                    batch_size, num_channels, height, width,
                    is_on_gpu
                )
            else
                # Backward para 2D mejorado
                backward_batchnorm_2d_improved!(
                    layer, input, grad_data,
                    saved_normalized, saved_mean, saved_std, saved_x,
                    num_features, batch_size,
                    is_on_gpu
                )
            end
        end
    end
    
    return out
end

# Función auxiliar mejorada para backward 4D
function backward_batchnorm_4d_improved!(layer, input, grad_output, x_norm, mean_save, std_save, x_save,
                                        N, C, H, W, is_on_gpu)
    # Preparar datos
    gamma_data = layer.gamma.data
    if is_on_gpu && !(gamma_data isa CUDA.CuArray)
        gamma_data = CUDA.CuArray(gamma_data)
    elseif !is_on_gpu && gamma_data isa CUDA.CuArray
        gamma_data = Array(gamma_data)
    end
    
    # Reshape para broadcasting
    gamma_reshaped = reshape(gamma_data, (1, C, 1, 1))
    std_reshaped = reshape(std_save, (1, C, 1, 1))
    mean_reshaped = reshape(mean_save, (1, C, 1, 1))
    
    # CRÍTICO: Solo calcular gradientes para parámetros que los necesitan
    if layer.gamma.requires_grad
        grad_gamma = sum(grad_output .* x_norm, dims=(1,3,4))[1,:,1,1]
        if grad_gamma isa CUDA.CuArray
            grad_gamma = Array(grad_gamma)
        end
        TensorEngine.backward(layer.gamma, TensorEngine.Tensor(grad_gamma; requires_grad=false))
    end
    
    if layer.beta.requires_grad
        grad_beta = sum(grad_output, dims=(1,3,4))[1,:,1,1]
        if grad_beta isa CUDA.CuArray
            grad_beta = Array(grad_beta)
        end
        TensorEngine.backward(layer.beta, TensorEngine.Tensor(grad_beta; requires_grad=false))
    end
    
    # Gradiente para input solo si es necesario
    if input.requires_grad
        if layer.training
            # Training mode: full gradient computation
            M = Float32(N * H * W)
            
            # Términos intermedios
            grad_xnorm = grad_output .* gamma_reshaped
            
            # Gradientes más estables
            term1 = grad_xnorm ./ std_reshaped
            term2 = sum(grad_xnorm .* x_norm, dims=(1,3,4)) ./ (M * std_reshaped)
            term3 = sum(grad_xnorm, dims=(1,3,4)) .* mean_reshaped ./ (M * std_reshaped)
            
            grad_input = term1 .- x_norm .* term2 .- term3
        else
            # Eval mode: simplified gradient
            grad_input = (grad_output .* gamma_reshaped) ./ std_reshaped
        end
        
        TensorEngine.backward(input, TensorEngine.Tensor(grad_input; requires_grad=false))
    end
end

# Función auxiliar mejorada para backward 2D
function backward_batchnorm_2d_improved!(layer, input, grad_output, x_norm, mean_save, std_save, x_save,
                                        F, B, is_on_gpu)
    # Similar a 4D pero para formato 2D
    gamma_data = layer.gamma.data
    if is_on_gpu && !(gamma_data isa CUDA.CuArray)
        gamma_data = CUDA.CuArray(gamma_data)
    elseif !is_on_gpu && gamma_data isa CUDA.CuArray
        gamma_data = Array(gamma_data)
    end
    
    gamma_reshaped = reshape(gamma_data, (F, 1))
    std_reshaped = reshape(std_save, (F, 1))
    mean_reshaped = reshape(mean_save, (F, 1))
    
    # Gradientes para parámetros
    if layer.gamma.requires_grad
        grad_gamma = sum(grad_output .* x_norm, dims=2)[:, 1]
        if grad_gamma isa CUDA.CuArray
            grad_gamma = Array(grad_gamma)
        end
        TensorEngine.backward(layer.gamma, TensorEngine.Tensor(grad_gamma; requires_grad=false))
    end
    
    if layer.beta.requires_grad
        grad_beta = sum(grad_output, dims=2)[:, 1]
        if grad_beta isa CUDA.CuArray
            grad_beta = Array(grad_beta)
        end
        TensorEngine.backward(layer.beta, TensorEngine.Tensor(grad_beta; requires_grad=false))
    end
    
    # Gradiente para input
    if input.requires_grad
        if layer.training
            M = Float32(B)
            grad_xnorm = grad_output .* gamma_reshaped
            
            term1 = grad_xnorm ./ std_reshaped
            term2 = sum(grad_xnorm .* x_norm, dims=2) ./ (M * std_reshaped)
            term3 = sum(grad_xnorm, dims=2) .* mean_reshaped ./ (M * std_reshaped)
            
            grad_input = term1 .- x_norm .* term2 .- term3
        else
            grad_input = (grad_output .* gamma_reshaped) ./ std_reshaped
        end
        
        TensorEngine.backward(input, TensorEngine.Tensor(grad_input; requires_grad=false))
    end
end

# Función auxiliar para backward 4D
function backward_batchnorm_4d!(layer, input, grad_output, x_norm, mean_save, std_save,
                                N, C, H, W, is_on_gpu)
    # Preparar datos
    gamma_data = layer.gamma.data
    if is_on_gpu && !(gamma_data isa CUDA.CuArray)
        gamma_data = CUDA.CuArray(gamma_data)
    elseif !is_on_gpu && gamma_data isa CUDA.CuArray
        gamma_data = Array(gamma_data)
    end
    
    # Reshape para broadcasting
    gamma_reshaped = reshape(gamma_data, (1, C, 1, 1))
    std_reshaped = reshape(std_save, (1, C, 1, 1))
    
    # Gradientes para gamma y beta
    grad_gamma = sum(grad_output .* x_norm, dims=(1,3,4))[1,:,1,1]
    grad_beta = sum(grad_output, dims=(1,3,4))[1,:,1,1]
    
    # Asegurar que estén en CPU para acumulación
    if grad_gamma isa CUDA.CuArray
        grad_gamma = Array(grad_gamma)
        grad_beta = Array(grad_beta)
    end
    
    # Gradiente para input
    if layer.training
        # Training mode: full gradient computation
        M = N * H * W  # número total de elementos por canal
        
        # Términos intermedios
        grad_xnorm = grad_output .* gamma_reshaped
        grad_var = sum(grad_xnorm .* x_norm .* (-0.5f0) ./ (std_reshaped .^ 3), dims=(1,3,4))
        grad_mean = sum(grad_xnorm ./ (-std_reshaped), dims=(1,3,4))
        
        # Input gradient
        grad_input = (grad_xnorm ./ std_reshaped) .+ 
                     (grad_var .* 2.0f0 .* (input.data .- reshape(mean_save, (1,C,1,1))) ./ M) .+
                     (grad_mean ./ M)
    else
        # Eval mode: simplified gradient
        grad_input = (grad_output .* gamma_reshaped) ./ std_reshaped
    end
    
    # Propagar gradientes
    TensorEngine.backward(layer.gamma, TensorEngine.Tensor(grad_gamma; requires_grad=false))
    TensorEngine.backward(layer.beta, TensorEngine.Tensor(grad_beta; requires_grad=false))
    TensorEngine.backward(input, TensorEngine.Tensor(grad_input; requires_grad=false))
end

# Función auxiliar para backward 2D
function backward_batchnorm_2d!(layer, input, grad_output, x_norm, mean_save, std_save,
                                F, B, is_on_gpu)
    # Similar a 4D pero para formato 2D
    gamma_data = layer.gamma.data
    if is_on_gpu && !(gamma_data isa CUDA.CuArray)
        gamma_data = CUDA.CuArray(gamma_data)
    elseif !is_on_gpu && gamma_data isa CUDA.CuArray
        gamma_data = Array(gamma_data)
    end
    
    gamma_reshaped = reshape(gamma_data, (F, 1))
    std_reshaped = reshape(std_save, (F, 1))
    
    # Gradientes para parámetros
    grad_gamma = sum(grad_output .* x_norm, dims=2)[:, 1]
    grad_beta = sum(grad_output, dims=2)[:, 1]
    
    if grad_gamma isa CUDA.CuArray
        grad_gamma = Array(grad_gamma)
        grad_beta = Array(grad_beta)
    end
    
    # Gradiente para input
    if layer.training
        grad_xnorm = grad_output .* gamma_reshaped
        grad_var = sum(grad_xnorm .* x_norm .* (-0.5f0) ./ (std_reshaped .^ 3), dims=2)
        grad_mean = sum(grad_xnorm ./ (-std_reshaped), dims=2)
        
        grad_input = (grad_xnorm ./ std_reshaped) .+ 
                     (grad_var .* 2.0f0 .* (input.data .- reshape(mean_save, (F,1))) ./ B) .+
                     (grad_mean ./ B)
    else
        grad_input = (grad_output .* gamma_reshaped) ./ std_reshaped
    end
    
    # Propagar gradientes
    TensorEngine.backward(layer.gamma, TensorEngine.Tensor(grad_gamma; requires_grad=false))
    TensorEngine.backward(layer.beta, TensorEngine.Tensor(grad_beta; requires_grad=false))
    TensorEngine.backward(input, TensorEngine.Tensor(grad_input; requires_grad=false))
end

# Hacer callable
function (layer::BatchNorm)(input::TensorEngine.Tensor)
    return forward(layer, input)
end

# Funciones de utilidad
function set_training!(layer::BatchNorm, training::Bool)
    layer.training = training
end

function reset_running_stats!(layer::BatchNorm)
    fill!(layer.running_mean, 0f0)
    fill!(layer.running_var, 1f0)
    layer.num_batches_tracked = 0
end

# Para debugging
function get_stats(layer::BatchNorm)
    return (
        mean = copy(layer.running_mean),
        var = copy(layer.running_var),
        num_batches = layer.num_batches_tracked,
        training = layer.training
    )
end

# ==================================================================
# Dropout Optimizado (Bitmask en GPU)
# ==================================================================
"""
Versión mejorada de Dropout que es callable directamente
"""
struct DropoutLayer <: AbstractLayer.Layer
    rate::Float32    # Tasa de dropout
    training::Bool   # Modo entrenamiento/inferencia
end

# Constructor con tipos Float32 explícitos
function DropoutLayer(rate::Real; training=true)
    return DropoutLayer(Float32(rate), training)
end

function forward(layer::DropoutLayer, input::TensorEngine.Tensor)
    if !layer.training || layer.rate == 0
        return input
    end
    
    # Crear máscara de dropout - Corregido para CUDA
    is_on_gpu = (input.data isa CUDA.CuArray)
    input_shape = size(input.data)
    
    if is_on_gpu
        # En GPU, necesitamos generar la máscara de manera segura
        # Crear un array de números aleatorios en GPU
        rand_vals = CUDA.rand(Float32, input_shape)
        # Crear máscara como array de Float32 (no BitArray)
        mask = CUDA.float(rand_vals .> layer.rate)  # Convertir a Float32 directamente
        # Aplicar máscara y escalar
        scale = 1.0f0 / (1.0f0 - layer.rate)  # Escalar como Float32
        output = input.data .* mask .* scale  # Multiplicación escalar en vez de división
    else
        # En CPU 
        rand_vals = rand(Float32, input_shape)
        mask = float(rand_vals .> layer.rate)
        scale = 1.0f0 / (1.0f0 - layer.rate)
        output = input.data .* mask .* scale
    end
    
    # Crear tensor de salida
    result = TensorEngine.Tensor(output)
    
    return result
end

# Hacer la capa callable directamente
function (layer::DropoutLayer)(input::TensorEngine.Tensor)
    return forward(layer, input)
end

# ==================================================================
# Flatten Optimizado (Soporte batches en formato WHCN)
# ==================================================================
struct Flatten <: AbstractLayer.Layer end

function (layer::Flatten)(input::TensorEngine.Tensor)
    dims = size(input.data)
    @assert length(dims) >= 2 "El tensor de entrada debe tener al menos 2 dimensiones."
    
    is_on_gpu = (input.data isa CUDA.CuArray)
    
    # Detección mejorada de formato basada en patrones comunes
    format = detect_format(dims)
    
    # Caso específico para salida de GlobalAvgPool (batch, channels, 1, 1)
    if length(dims) == 4 && dims[3] == 1 && dims[4] == 1
        batch_size = dims[1]
        channels = dims[2]
        
        if is_on_gpu
            reshaped = CUDA.reshape(input.data, (batch_size, channels))
            new_data = CUDA.permutedims(reshaped, (2, 1))  # (channels, batch)
        else
            reshaped = reshape(input.data, (batch_size, channels))
            new_data = permutedims(reshaped, (2, 1))  # (channels, batch)
        end
    else
        if format == :NCHW
            # Formato NCHW: (batch, canales, alto, ancho)
            n = dims[1]
            flat_dim = prod(dims[2:end])
            
            # Aplanar a (flat_dim, n) para Dense
            if is_on_gpu
                reshaped = CUDA.reshape(input.data, (n, flat_dim))
                new_data = CUDA.permutedims(reshaped, (2, 1))
            else
                reshaped = reshape(input.data, (n, flat_dim))
                new_data = permutedims(reshaped, (2, 1))
            end
        else  # format == :WHCN
            # Formato WHCN (original): (W, H, C, N)
            flat_dim = prod(dims[1:end-1])
            batch_dim = dims[end]
            
            # Aplanamos a (flat_dim, batch_dim)
            if is_on_gpu
                new_data = CUDA.reshape(input.data, (flat_dim, batch_dim))
            else
                new_data = reshape(input.data, (flat_dim, batch_dim))
            end
        end
    end
    
    out = TensorEngine.Tensor(new_data)
    
    # Implementación mejorada de backward
    out.backward_fn = grad -> begin
        grad_data = grad
        
        # Asegurar mismo dispositivo que input
        if (grad_data isa CUDA.CuArray) != is_on_gpu
            if is_on_gpu
                grad_data = CUDA.CuArray(grad_data)
            else
                grad_data = Array(grad_data)
            end
        end
        
        # Restaurar forma original según el caso
        if length(dims) == 4 && dims[3] == 1 && dims[4] == 1
            batch_size = dims[1]
            channels = dims[2]
            
            if is_on_gpu
                grad_reshaped = CUDA.permutedims(grad_data, (2, 1))
                restored = CUDA.reshape(grad_reshaped, (batch_size, channels, 1, 1))
            else
                grad_reshaped = permutedims(grad_data, (2, 1))
                restored = reshape(grad_reshaped, (batch_size, channels, 1, 1))
            end
        elseif format == :NCHW
            if is_on_gpu
                grad_reshaped = CUDA.permutedims(grad_data, (2, 1))
                restored = CUDA.reshape(grad_reshaped, dims)
            else
                grad_reshaped = permutedims(grad_data, (2, 1))
                restored = reshape(grad_reshaped, dims)
            end
        else  # format == :WHCN
            if is_on_gpu
                restored = CUDA.reshape(grad_data, dims)
            else
                restored = reshape(grad_data, dims)
            end
        end
        
        TensorEngine.backward(input, TensorEngine.Tensor(restored))
    end
    
    return out
end

# Función auxiliar para detectar formato
function detect_format(dims::Tuple)
    if length(dims) != 4
        return :UNKNOWN
    end
    
    # Heurísticas para detectar formato:
    # NCHW: batch suele ser pequeño (1-256), canales moderados (3-2048)
    # WHCN: batch al final, dimensiones espaciales primero
    
    # Si la primera dimensión es pequeña y la segunda parece canales
    if dims[1] <= 256 && dims[2] in [1, 3, 16, 32, 64, 128, 256, 512, 1024, 2048]
        return :NCHW
    # Si las primeras dos dimensiones son grandes (espaciales) y la última es pequeña (batch)
    elseif dims[1] > 10 && dims[2] > 10 && dims[4] <= 256
        return :WHCN
    else
        # Default a NCHW si no está claro
        return :NCHW
    end
end

# ==================================================================
# AdaptiveAvgPool y GlobalAvgPool
# ==================================================================
struct AdaptiveAvgPool <: AbstractLayer.Layer
    output_size::Tuple{Int,Int}
end

function forward(layer::AdaptiveAvgPool, input::TensorEngine.Tensor)
    output = NNlib.adaptive_avg_pool(input.data, layer.output_size)
    return TensorEngine.Tensor(output)
end

# ==================================================================
# GlobalAvgPool modificado con depuración
# ==================================================================
struct GlobalAvgPool <: AbstractLayer.Layer end

function forward(layer::GlobalAvgPool, input::TensorEngine.Tensor)
    # Añadir depuración
    #println("GlobalAvgPool - Input shape: ", size(input.data))
    
    dims = size(input.data)
    if length(dims) == 4  # Tensor 4D (NCHW)
        # Promedio en dimensiones espaciales (alto y ancho)
        output = mean(input.data, dims=(3, 4))
    else
        # Manejo de caso general
        output = input.data
    end
    
    #println("GlobalAvgPool - Output shape: ", size(output))
    return TensorEngine.Tensor(output)
end

# Método para hacer GlobalAvgPool callable
function (layer::GlobalAvgPool)(input::TensorEngine.Tensor)
    return forward(layer, input)
end


# CustomFlatten adaptado al formato que espera Dense
struct CustomFlatten <: AbstractLayer.Layer end

function (layer::CustomFlatten)(input::TensorEngine.Tensor)
    data = input.data
    dims = size(data)
    
    println("CustomFlatten - Entrada: ", dims)
    
    if length(dims) == 4
        # Para formato NCHW (batch, canales, alto, ancho)
        n, c, h, w = dims
        
        # Aplanar todos los píxeles
        flat_features = c * h * w
        
        # Primero aplanar manteniendo batch
        reshaped = reshape(data, (n, flat_features))
        
        # Luego transponer para tener (features, batch)
        transposed = permutedims(reshaped, (2, 1))
        
        println("CustomFlatten - Salida: ", size(transposed))
        
        return TensorEngine.Tensor(transposed)
    elseif length(dims) == 3
        # Para tensores 3D (canales, alto, ancho)
        c, h, w = dims
        flat_features = c * h * w
        
        # Reformatear a (features, 1)
        reshaped = reshape(data, (flat_features, 1))
        
        return TensorEngine.Tensor(reshaped)
    else
        # Para otros casos, simplemente retornar
        return input
    end
end


struct LayerActivation <: AbstractLayer.Layer
    f::Function
end

function (a::LayerActivation)(input::TensorEngine.Tensor)
    return a.f(input)
end

# Redefinir relu si es necesario
function relu(t::TensorEngine.Tensor)
    data_out = max.(t.data, 0.0f0)
    out = TensorEngine.Tensor(data_out)
    out.backward_fn = grad -> begin
        TensorEngine.backward(t, TensorEngine.Tensor((t.data .> 0) .* grad))
    end
    return out
end



# ==================================================================
# Bloques Residuales
# ==================================================================
"""
Implementación de bloques residuales para mejor flujo de gradientes
"""

struct ResidualBlock <: AbstractLayer.Layer
    conv_path::Vector{<:AbstractLayer.Layer}
    shortcut::Vector{<:AbstractLayer.Layer}
end



# En Layers.jl - Mejorar el debug en ResidualBlock
function forward(block::ResidualBlock, input::TensorEngine.Tensor)
    # Debug para verificar dimensiones
    input_shape = size(input.data)
    println("ResidualBlock input shape: $input_shape")
    
    # Camino convolucional
    conv_out = input
    for (i, layer) in enumerate(block.conv_path)
        conv_out = layer(conv_out)
        println("  Conv path layer $i output shape: $(size(conv_out.data))")
    end
    
    # Camino shortcut
    shortcut_out = input
    if !isempty(block.shortcut)
        for (i, layer) in enumerate(block.shortcut)
            shortcut_out = layer(shortcut_out)
            println("  Shortcut layer $i output shape: $(size(shortcut_out.data))")
        end
    end
    
    # Verificar que las dimensiones coincidan antes de sumar
    conv_shape = size(conv_out.data)
    shortcut_shape = size(shortcut_out.data)
    
    if conv_shape != shortcut_shape
        error("Shape mismatch in ResidualBlock: conv_path=$conv_shape, shortcut=$shortcut_shape")
    end
    
    # Sumar ambos caminos
    output = TensorEngine.add(conv_out, shortcut_out)
    println("  ResidualBlock output shape: $(size(output.data))")
    
    # Aplicar activación final
    return LayerActivation(relu)(output)
end

function (block::ResidualBlock)(input::TensorEngine.Tensor)
    return forward(block, input)
end

# Función auxiliar para crear bloques residuales
function create_residual_block(in_channels, out_channels, stride=1)
    conv_path = AbstractLayer.Layer[
        ConvKernelLayer(in_channels, out_channels, (3,3), stride=(stride,stride), padding=(1,1)),
        BatchNorm(out_channels),
        LayerActivation(relu),  # Usar LayerActivation en lugar de Activation
        ConvKernelLayer(out_channels, out_channels, (3,3), stride=(1,1), padding=(1,1)),
        BatchNorm(out_channels)
    ]
    
    if stride != 1 || in_channels != out_channels
        # Si hay cambio de dimensión, necesitamos proyección en el shortcut
        shortcut = AbstractLayer.Layer[
            ConvKernelLayer(in_channels, out_channels, (1,1), stride=(stride,stride), padding=(0,0)),
            BatchNorm(out_channels)
        ]
    else
        shortcut = AbstractLayer.Layer[]
    end
    
    return ResidualBlock(conv_path, shortcut)
end



end  # module Layers