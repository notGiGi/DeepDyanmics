module Layers

using NNlib, CUDA, ..TensorEngine, ..AbstractLayer, ..ConvKernelLayers, Statistics, ..ConvolutionalLayers
export BatchNorm, Flatten, LayerActivation, set_training!, reset_running_stats!, GlobalAvgPool, ResidualBlock, create_residual_block, CustomFlatten, DropoutLayer,
       LayerNorm, RNNCell, RNN, LSTMCell, LSTM, GRUCell, GRU


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
"Acumula `g` en `param.grad` respetando dispositivo y tipo"
function _accum_grad!(param::TensorEngine.Tensor, g)
    # 1.  Mover g al mismo dispositivo que param.data
    if (param.data isa CUDA.CuArray) != (g isa CUDA.CuArray)
        g = param.data isa CUDA.CuArray ? CUDA.CuArray(g) : Array(g)
    end

    # 2.  Crear o acumular
    if param.grad === nothing
        param.grad = TensorEngine.Tensor(g; requires_grad = false)
    else
        param.grad.data .+= g
    end
end

# backward mejorado ‑ 4‑D  (N,C,H,W)
# ---------------------------------------------------------------------
function backward_batchnorm_4d_improved!(layer, input, grad_output, x_norm,
                                         mean_save, std_save, x_save,
                                         N, C, H, W, is_on_gpu)

    # Aseguramos que gamma está en el mismo dispositivo que grad_output
    gamma_data = layer.gamma.data
    if is_on_gpu && !(gamma_data isa CUDA.CuArray)
        gamma_data = CUDA.CuArray(gamma_data)
    elseif !is_on_gpu &&  (gamma_data isa CUDA.CuArray)
        gamma_data = Array(gamma_data)
    end

    gamma_reshaped = reshape(gamma_data, (1,C,1,1))
    std_reshaped   = reshape(std_save,  (1,C,1,1))
    mean_reshaped  = reshape(mean_save, (1,C,1,1))

    # -------- gradientes para gamma y beta ---------------------------
    if layer.gamma.requires_grad
        grad_gamma = sum(grad_output .* x_norm, dims=(1,3,4))[1,:,1,1]  # (4D)
        _accum_grad!(layer.gamma, Array(grad_gamma))
    end

    if layer.beta.requires_grad
        grad_beta = sum(grad_output, dims=(1,3,4))[1,:,1,1]             # (4D)
        _accum_grad!(layer.beta, Array(grad_beta))
    end

    # -------- gradiente para la entrada (solo si se pide) ------------
    if input.requires_grad
        if layer.training
            M = Float32(N * H * W)
            grad_xnorm = grad_output .* gamma_reshaped
            term1 = grad_xnorm ./ std_reshaped
            term2 = sum(grad_xnorm .* x_norm; dims=(1,3,4)) ./ (M * std_reshaped)
            term3 = sum(grad_xnorm;            dims=(1,3,4)) .* mean_reshaped ./ (M * std_reshaped)
            grad_input = term1 .- x_norm .* term2 .- term3
        else
            grad_input = (grad_output .* gamma_reshaped) ./ std_reshaped
        end
        TensorEngine.backward(input,
                              TensorEngine.Tensor(grad_input; requires_grad=false))
    end
end


# ---------------------------------------------------------------------
# backward mejorado ‑ 2‑D  (F,B)  — para capas Dense
# ---------------------------------------------------------------------
function backward_batchnorm_2d_improved!(layer, input, grad_output, x_norm,
                                         mean_save, std_save, x_save,
                                         F, B, is_on_gpu)

    gamma_data = layer.gamma.data
    if is_on_gpu && !(gamma_data isa CUDA.CuArray)
        gamma_data = CUDA.CuArray(gamma_data)
    elseif !is_on_gpu &&  (gamma_data isa CUDA.CuArray)
        gamma_data = Array(gamma_data)
    end

    gamma_reshaped = reshape(gamma_data, (F,1))
    std_reshaped   = reshape(std_save,  (F,1))
    mean_reshaped  = reshape(mean_save, (F,1))

    # -------- gradientes para gamma y beta ---------------------------
    if layer.gamma.requires_grad
        grad_gamma = sum(grad_output .* x_norm, dims=2)[:,1]
        _accum_grad!(layer.gamma, Array(grad_gamma))
    end
    if layer.beta.requires_grad
        grad_beta = sum(grad_output, dims=2)[:,1]
        _accum_grad!(layer.beta, Array(grad_beta))
    end


    # -------- gradiente para la entrada ------------------------------
    if input.requires_grad
        if layer.training
            M = Float32(B)
            grad_xnorm = grad_output .* gamma_reshaped
            term1 = grad_xnorm ./ std_reshaped
            term2 = sum(grad_xnorm .* x_norm; dims=2) ./ (M * std_reshaped)
            term3 = sum(grad_xnorm;            dims=2) .* mean_reshaped ./ (M * std_reshaped)
            grad_input = term1 .- x_norm .* term2 .- term3
        else
            grad_input = (grad_output .* gamma_reshaped) ./ std_reshaped
        end
        TensorEngine.backward(input,
                              TensorEngine.Tensor(grad_input; requires_grad=false))
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
# Definición de DropoutLayer
mutable struct DropoutLayer <: AbstractLayer.Layer
    rate::Float32
    training::Bool
end

# Constructor
function DropoutLayer(rate::Real; training=true)
    return DropoutLayer(Float32(rate), training)
end

# IMPLEMENTACIÓN COMPLETA DE FORWARD
"""
    forward(layer::DropoutLayer, input::TensorEngine.Tensor)

Aplica dropout durante entrenamiento, pasa sin cambios durante evaluación.
Mantiene la media esperada usando inverted dropout.
"""
function forward(layer::DropoutLayer, input::TensorEngine.Tensor)
    # Sin dropout en modo evaluación o si rate es 0
    if !layer.training || layer.rate == 0
        return input
    end
    
    # Validar tasa de dropout
    if !(0 <= layer.rate < 1)
        throw(ArgumentError("Dropout rate debe estar en [0, 1), recibido: $(layer.rate)"))
    end
    
    # Detectar dispositivo
    is_on_gpu = input.data isa CUDA.CuArray
    
    # Escala para inverted dropout
    scale = 1.0f0 / (1.0f0 - layer.rate)
    
    # Generar máscara binaria
    if is_on_gpu
        # Para GPU
        rand_vals = CUDA.rand(Float32, size(input.data))
        mask = rand_vals .> layer.rate
        output_data = input.data .* mask .* scale
    else
        # Para CPU
        rand_vals = rand(Float32, size(input.data))
        mask = rand_vals .> layer.rate
        output_data = input.data .* mask .* scale
    end
    
    # Crear tensor de salida con requires_grad correcto
    out = TensorEngine.Tensor(output_data; requires_grad=input.requires_grad)
    
    # Definir función backward solo si necesario
    if input.requires_grad
        # Capturar variables para el closure
        saved_mask = mask
        saved_scale = scale
        saved_is_gpu = is_on_gpu
        
        out.backward_fn = function(grad)
            # Convertir grad si es necesario
            grad_data = grad isa TensorEngine.Tensor ? grad.data : grad
            
            # Asegurar dispositivo correcto
            if (grad_data isa CUDA.CuArray) != saved_is_gpu
                if saved_is_gpu
                    grad_data = CUDA.CuArray(grad_data)
                else
                    grad_data = Array(grad_data)
                end
            end
            
            # Propagar gradiente con la misma máscara
            grad_input = grad_data .* saved_mask .* saved_scale
            
            # Llamar backward del input
            TensorEngine.backward(input, TensorEngine.Tensor(grad_input; requires_grad=false))
        end
    end
    
    return out
end

# Hacer DropoutLayer callable (azúcar sintáctico)
function (layer::DropoutLayer)(input::TensorEngine.Tensor)
    return forward(layer, input)
end



# ==================================================================
# Flatten Optimizado (Soporte batches en formato WHCN)
# ==================================================================
struct Flatten <: AbstractLayer.Layer end

# Reemplazar la función (layer::Flatten) completa con esta versión que incluye el caso 2D:

function (layer::Flatten)(input::TensorEngine.Tensor)
    A   = input.data
    nd  = ndims(A)
    dev_gpu = A isa CUDA.CuArray

    # 1) 1D -> (D,1)
    if nd == 1
        out_data = reshape(A, (length(A), 1))
        out = TensorEngine.Tensor(out_data; requires_grad=input.requires_grad)
        if out.requires_grad
            out.backward_fn = g -> begin
                G = g isa TensorEngine.Tensor ? g.data : g           # (D,1)
                gx = reshape(G, (length(A),))
                TensorEngine.backward(input, TensorEngine.Tensor(gx))
            end
        end
        return out
    end

    # 2) 2D -> identidad (D,N)
    if nd == 2
        out = TensorEngine.Tensor(A; requires_grad=input.requires_grad)
        if out.requires_grad
            out.backward_fn = g -> TensorEngine.backward(input, g isa TensorEngine.Tensor ? g : TensorEngine.Tensor(g))
        end
        return out
    end

    # 3) 3D -> caso secuencias (T,N,E) => (E*T, N)   *** Embedding nuevo ***
    if nd == 3
        T, N, E = size(A)
        # (T,N,E) -> (E,T,N)
        Ap = permutedims(A, (3, 1, 2))
        # (E,T,N) -> (E*T, N)
        out_data = reshape(Ap, (E*T, N))
        out = TensorEngine.Tensor(out_data; requires_grad=input.requires_grad)

        if out.requires_grad
            out.backward_fn = g -> begin
                G  = g isa TensorEngine.Tensor ? g.data : g          # (E*T, N)
                G3 = reshape(G, (E, T, N))                           # (E,T,N)
                Gorig = permutedims(G3, (2, 3, 1))                   # (T,N,E)
                TensorEngine.backward(input, TensorEngine.Tensor(Gorig))
            end
        end
        return out
    end

    # 4) ≥4D: soportar NCHW típico y GAP (N,C,1,1) preservando batch
    dims = size(A)

    # (N,C,1,1) → (C,N)
    if nd == 4 && dims[3] == 1 && dims[4] == 1
        N, C = dims[1], dims[2]
        resh = reshape(A, (N, C))
        out_data = permutedims(resh, (2, 1))                          # (C,N)
        out = TensorEngine.Tensor(out_data; requires_grad=input.requires_grad)
        if out.requires_grad
            out.backward_fn = g -> begin
                G = g isa TensorEngine.Tensor ? g.data : g            # (C,N)
                Gr = permutedims(G, (2, 1))                           # (N,C)
                G4 = reshape(Gr, (N, C, 1, 1))
                TensorEngine.backward(input, TensorEngine.Tensor(G4))
            end
        end
        return out
    end

    # Fallback ≥4D: asumir batch en primer eje (N, ...) → (prod(resto), N)
    N = dims[1]
    feat = Int(prod(dims[2:end]))
    resh = reshape(A, (N, feat))                                      # (N,feat)
    out_data = permutedims(resh, (2, 1))                               # (feat,N)
    out = TensorEngine.Tensor(out_data; requires_grad=input.requires_grad)

    if out.requires_grad
        out.backward_fn = g -> begin
            G = g isa TensorEngine.Tensor ? g.data : g                # (feat,N)
            Gr = permutedims(G, (2, 1))                               # (N,feat)
            Gorig = reshape(Gr, dims)
            TensorEngine.backward(input, TensorEngine.Tensor(Gorig))
        end
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
    # Input shape: (N, C, H, W)
    N, C, H, W = size(input.data)
    
    # Average over spatial dimensions
    output_data = mean(input.data, dims=(3,4))  # Result: (N, C, 1, 1)
    output = TensorEngine.Tensor(output_data)
    
    # CRÍTICO: Agregar backward_fn
    output.backward_fn = grad -> begin
        # El gradiente se distribuye uniformemente sobre H*W elementos
        grad_data = grad isa TensorEngine.Tensor ? grad.data : grad
        
        # Expandir el gradiente a las dimensiones originales
        # grad viene como (N, C, 1, 1), necesitamos (N, C, H, W)
        grad_expanded = grad_data ./ (H * W)  # Dividir por número de elementos promediados
        
        # Broadcast a las dimensiones originales
        grad_input = repeat(grad_expanded, 1, 1, H, W)
        
        TensorEngine.backward(input, TensorEngine.Tensor(grad_input))
    end
    
    return output
end

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



function forward(block::ResidualBlock, input::TensorEngine.Tensor)
    # Camino convolucional
    conv_out = input
    for layer in block.conv_path
        conv_out = layer(conv_out)
    end
    
    # Camino shortcut
    shortcut_out = input
    if !isempty(block.shortcut)
        for layer in block.shortcut
            shortcut_out = layer(shortcut_out)
        end
    end
    
    # Verificar que las dimensiones coincidan
    conv_shape = size(conv_out.data)
    shortcut_shape = size(shortcut_out.data)

    
    if conv_shape != shortcut_shape
        error("""
        Shape mismatch in ResidualBlock:
        - Conv path output: $conv_shape
        - Shortcut output: $shortcut_shape
        
        This typically happens when:
        1. Stride is not properly handled in shortcut
        2. Channel dimensions don't match
        3. Spatial dimensions are incompatible
        """)
    end
    
    # Sumar ambos caminos
    output = TensorEngine.add(conv_out, shortcut_out)
    
    # Aplicar activación final
    return LayerActivation(relu)(output)
end
function (block::ResidualBlock)(input::TensorEngine.Tensor)
    return forward(block, input)
end

# Función auxiliar para crear bloques residuales
function create_residual_block(in_channels, out_channels, stride=1)
    conv_path = AbstractLayer.Layer[
        Conv2D(in_channels, out_channels, (3,3); stride=(stride,stride), padding=(1,1)),
        BatchNorm(out_channels),
        LayerActivation(relu),
        Conv2D(out_channels, out_channels, (3,3); stride=(1,1), padding=(1,1)),
        BatchNorm(out_channels)
    ]
    
    if stride != 1 || in_channels != out_channels
        shortcut = AbstractLayer.Layer[
            Conv2D(in_channels, out_channels, (1,1); stride=(stride,stride), padding=(0,0)),
            BatchNorm(out_channels)
        ]
    else
        shortcut = AbstractLayer.Layer[]
    end
    
    return ResidualBlock(conv_path, shortcut)
end



mutable struct LayerNorm <: AbstractLayer.Layer
    normalized_shape::Tuple{Vararg{Int}}
    gamma::TensorEngine.Tensor
    beta::TensorEngine.Tensor
    eps::Float32
    training::Bool
end

function LayerNorm(normalized_shape::Union{Int, Tuple{Vararg{Int}}}; 
                   eps::Float32=1f-5, training::Bool=true)
    shape = normalized_shape isa Int ? (normalized_shape,) : normalized_shape
    
    gamma = TensorEngine.Tensor(ones(Float32, shape...); requires_grad=true)
    beta = TensorEngine.Tensor(zeros(Float32, shape...); requires_grad=true)
    
    return LayerNorm(shape, gamma, beta, eps, training)
end

function forward(ln::LayerNorm, x::TensorEngine.Tensor)
    x_shape = size(x.data)
    ndims_x = ndims(x.data)
    # --- Device guard: asegurar que gamma/beta están en el mismo dispositivo que x ---
    is_x_gpu = x.data isa CUDA.CuArray
    gamma_on_gpu = ln.gamma.data isa CUDA.CuArray
    beta_on_gpu  = ln.beta.data  isa CUDA.CuArray

    if is_x_gpu && (!gamma_on_gpu || !beta_on_gpu)
        ln.gamma = TensorEngine.to_gpu(ln.gamma)
        ln.beta  = TensorEngine.to_gpu(ln.beta)
    elseif !is_x_gpu && (gamma_on_gpu || beta_on_gpu)
        ln.gamma = TensorEngine.to_cpu(ln.gamma)
        ln.beta  = TensorEngine.to_cpu(ln.beta)
    end
    # Determinar dimensiones de normalización basado en el formato
    if ndims_x == 2  # Formato (features, batch)
        # Normalizar sobre features (dimensión 1)
        if x_shape[1] != ln.normalized_shape[1]
            error("Shape mismatch: expected features=$(ln.normalized_shape[1]), got $(x_shape[1])")
        end
        
        # Calcular estadísticas por sample (sobre features)
        mean_x = mean(x.data; dims=1)  # Sin keepdims
        var_x = var(x.data; dims=1, corrected=false)
        
        # Normalizar
        x_normalized = (x.data .- mean_x) ./ sqrt.(var_x .+ ln.eps)
        
        # Aplicar scale y shift
        # gamma y beta son (features,), necesitamos (features, 1) para broadcasting
        gamma_reshaped = reshape(ln.gamma.data, :, 1)
        beta_reshaped = reshape(ln.beta.data, :, 1)
        
    elseif ndims_x == 4  # Formato NCHW
        # Para NCHW, normalizar sobre las últimas dimensiones según normalized_shape
        n_norm_dims = length(ln.normalized_shape)
        
        if n_norm_dims == 1  # Normalizar solo sobre channels
            if x_shape[2] != ln.normalized_shape[1]
                error("Shape mismatch: expected channels=$(ln.normalized_shape[1]), got $(x_shape[2])")
            end
            
            # Normalizar sobre canales para cada posición espacial
            mean_x = mean(x.data; dims=2)  # Sin keepdims
            var_x = var(x.data; dims=2, corrected=false)
            
            # gamma y beta son (C,), reshape a (1, C, 1, 1)
            gamma_reshaped = reshape(ln.gamma.data, 1, :, 1, 1)
            beta_reshaped = reshape(ln.beta.data, 1, :, 1, 1)
            
        elseif n_norm_dims == 3  # Normalizar sobre (C, H, W)
            expected = (x_shape[2], x_shape[3], x_shape[4])
            if expected != ln.normalized_shape
                error("Shape mismatch: expected $(ln.normalized_shape), got $expected")
            end
            
            # Normalizar sobre las últimas 3 dimensiones
            mean_x = mean(x.data; dims=(2,3,4))  # Sin keepdims
            var_x = var(x.data; dims=(2,3,4), corrected=false)
            
            # gamma y beta son (C, H, W), reshape a (1, C, H, W)
            gamma_reshaped = reshape(ln.gamma.data, 1, size(ln.gamma.data)...)
            beta_reshaped = reshape(ln.beta.data, 1, size(ln.beta.data)...)
        else
            error("Invalid normalized_shape for 4D input: $(ln.normalized_shape)")
        end
        
        x_normalized = (x.data .- mean_x) ./ sqrt.(var_x .+ ln.eps)
        
    else
        error("LayerNorm only supports 2D (features, batch) or 4D (NCHW) inputs, got $(ndims_x)D")
    end
    
    # Aplicar transformación afín
    y = x_normalized .* gamma_reshaped .+ beta_reshaped
    
    out = TensorEngine.Tensor(y; requires_grad=x.requires_grad)
    
        if out.requires_grad
            # Guardar para backward
            saved_normalized = x_normalized
            saved_var = var_x
            saved_mean = mean_x
            saved_gamma_reshaped = gamma_reshaped
            
            out.backward_fn = function(grad)
        # Asegurar que grad esté en el dispositivo correcto
        grad_data = grad isa TensorEngine.Tensor ? grad.data : grad
        
        # Detectar dispositivo
        is_on_gpu = grad_data isa CUDA.CuArray
        
        # Gradientes para gamma y beta
        if ndims_x == 2
            grad_gamma = vec(sum(grad_data .* saved_normalized; dims=2))
            grad_beta = vec(sum(grad_data; dims=2))
            norm_dims = 1
        elseif ndims_x == 4 && length(ln.normalized_shape) == 1
            grad_gamma = vec(sum(grad_data .* saved_normalized; dims=(1,3,4)))
            grad_beta = vec(sum(grad_data; dims=(1,3,4)))
            norm_dims = 2
        else  # 4D con normalización completa
            grad_gamma = dropdims(sum(grad_data .* saved_normalized; dims=1); dims=1)
            grad_beta = dropdims(sum(grad_data; dims=1); dims=1)
            norm_dims = (2,3,4)
        end
        
        # Acumular gradientes - CORREGIDO para GPU
        if ln.gamma.grad === nothing
            if is_on_gpu
                ln.gamma.grad = TensorEngine.Tensor(CUDA.zeros(Float32, size(ln.gamma.data)))
            else
                ln.gamma.grad = TensorEngine.Tensor(zeros(Float32, size(ln.gamma.data)))
            end
        end
        if ln.beta.grad === nothing
            if is_on_gpu
                ln.beta.grad = TensorEngine.Tensor(CUDA.zeros(Float32, size(ln.beta.data)))
            else
                ln.beta.grad = TensorEngine.Tensor(zeros(Float32, size(ln.beta.data)))
            end
        end
        
        # Asegurar que los gradientes acumulados estén en el mismo dispositivo
        if is_on_gpu
            # Si estamos en GPU, asegurar que gamma.grad y beta.grad también lo estén
            if !(ln.gamma.grad.data isa CUDA.CuArray)
                ln.gamma.grad = TensorEngine.Tensor(CUDA.CuArray(ln.gamma.grad.data))
            end
            if !(ln.beta.grad.data isa CUDA.CuArray)
                ln.beta.grad = TensorEngine.Tensor(CUDA.CuArray(ln.beta.grad.data))
            end
            
            # Ahora acumular (todo en GPU)
            grad_gamma_reshaped = CUDA.reshape(grad_gamma, size(ln.gamma.data))
            grad_beta_reshaped = CUDA.reshape(grad_beta, size(ln.beta.data))
        else
            # En CPU
            grad_gamma_reshaped = reshape(grad_gamma, size(ln.gamma.data))
            grad_beta_reshaped = reshape(grad_beta, size(ln.beta.data))
        end
        
        # Acumular
        ln.gamma.grad.data .+= grad_gamma_reshaped
        ln.beta.grad.data .+= grad_beta_reshaped
        
        # Gradiente para x (resto del código igual)
        N = prod(size(x.data)) ÷ prod(size(saved_mean))
        
        grad_normalized = grad_data .* saved_gamma_reshaped
        std_inv = 1f0 ./ sqrt.(saved_var .+ ln.eps)
        
        # Calcular gradiente de x
        if ndims_x == 2
            grad_var = sum(grad_normalized .* (x.data .- saved_mean) .* (-0.5f0) .* (std_inv .^ 3f0); dims=norm_dims)
            grad_mean = sum(grad_normalized .* (-std_inv); dims=norm_dims)
            grad_x = (grad_normalized .* std_inv) .+ 
                    (grad_var .* 2f0 .* (x.data .- saved_mean) ./ N) .+ 
                    (grad_mean ./ N)
        else
            # Para 4D, manejar las dimensiones correctamente
            grad_var = sum(grad_normalized .* (x.data .- saved_mean) .* (-0.5f0) .* (std_inv .^ 3f0); dims=norm_dims)
            grad_mean = sum(grad_normalized .* (-std_inv); dims=norm_dims)
            grad_x = (grad_normalized .* std_inv) .+ 
                    (grad_var .* 2f0 .* (x.data .- saved_mean) ./ N) .+ 
                    (grad_mean ./ N)
        end
        
        TensorEngine.backward(x, TensorEngine.Tensor(grad_x; requires_grad=false))
    end
    end
    
    return out
end

# Hacer LayerNorm callable
function (ln::LayerNorm)(input::TensorEngine.Tensor)
    return forward(ln, input)
end

function set_training!(ln::LayerNorm, mode::Bool)
    ln.training = mode
    return ln
end

# ==================================================================
# RNN - Recurrent Neural Network
# (tu RNNCell queda igual; solo copio/pego para contexto)
# ==================================================================
mutable struct RNNCell <: AbstractLayer.Layer
    input_size::Int
    hidden_size::Int
    W_ih::TensorEngine.Tensor
    W_hh::TensorEngine.Tensor
    b_ih::Union{TensorEngine.Tensor, Nothing}
    b_hh::Union{TensorEngine.Tensor, Nothing}
    activation::Function
    training::Bool
end

function RNNCell(input_size::Int, hidden_size::Int;
                 bias::Bool=true, activation::Function=tanh,
                 requires_grad::Bool=true)
    σ = sqrt(2f0 / (input_size + hidden_size))

    W_ih = TensorEngine.Tensor(randn(Float32, hidden_size, input_size) .* σ;
                               requires_grad=requires_grad)
    W_hh = TensorEngine.Tensor(randn(Float32, hidden_size, hidden_size) .* σ;
                               requires_grad=requires_grad)

    b_ih = bias ? TensorEngine.Tensor(zeros(Float32, hidden_size, 1);
                                      requires_grad=requires_grad) : nothing
    b_hh = bias ? TensorEngine.Tensor(zeros(Float32, hidden_size, 1);
                                      requires_grad=requires_grad) : nothing

    return RNNCell(input_size, hidden_size, W_ih, W_hh, b_ih, b_hh, activation, true)
end

function forward(cell::RNNCell, input::TensorEngine.Tensor,
                 hidden::Union{TensorEngine.Tensor, Nothing}=nothing)
    batch_size = size(input.data, 2)
    device = TensorEngine.device_of(input)

    if hidden === nothing
        h_data = zeros(Float32, cell.hidden_size, batch_size)
        h_data = device == :gpu ? CUDA.cu(h_data) : h_data
        hidden = TensorEngine.Tensor(h_data; requires_grad=false)
    end

    W_ih_data = TensorEngine.ensure_on_device(cell.W_ih.data, device)
    W_hh_data = TensorEngine.ensure_on_device(cell.W_hh.data, device)
    h = hidden.data
    x = input.data

    output = W_ih_data * x + W_hh_data * h

    if cell.b_ih !== nothing
        b_ih_data = TensorEngine.ensure_on_device(cell.b_ih.data, device)
        output = output .+ b_ih_data
    end
    if cell.b_hh !== nothing
        b_hh_data = TensorEngine.ensure_on_device(cell.b_hh.data, device)
        output = output .+ b_hh_data
    end

    pre_activation = copy(output)

    if cell.activation == tanh
        output = tanh.(output)
        activation_grad_fn = grad -> grad .* (1f0 .- output.^2)
    elseif cell.activation == relu
        output = max.(output, 0f0)
        activation_grad_fn = grad -> grad .* (pre_activation .> 0f0)
    elseif cell.activation == sigmoid
        output = 1f0 ./ (1f0 .+ exp.(-output))
        activation_grad_fn = grad -> grad .* output .* (1f0 .- output)
    else
        output = cell.activation(output)
        activation_grad_fn = grad -> grad
    end

    h_new = TensorEngine.Tensor(output;
                                requires_grad=input.requires_grad || cell.W_ih.requires_grad)

    if h_new.requires_grad
        h_new.backward_fn = function(grad)
            grad_data = grad isa TensorEngine.Tensor ? grad.data : grad
            grad_act = activation_grad_fn(grad_data)

            ∇W_ih = grad_act * x'
            ∇W_hh = grad_act * h'

            if cell.b_ih !== nothing && cell.b_ih.requires_grad
                ∇b_ih = sum(grad_act, dims=2)
                TensorEngine.backward(cell.b_ih, TensorEngine.Tensor(∇b_ih; requires_grad=false))
            end
            if cell.b_hh !== nothing && cell.b_hh.requires_grad
                ∇b_hh = sum(grad_act, dims=2)
                TensorEngine.backward(cell.b_hh, TensorEngine.Tensor(∇b_hh; requires_grad=false))
            end

            ∇x = W_ih_data' * grad_act
            ∇h = W_hh_data' * grad_act

            if cell.W_ih.requires_grad
                TensorEngine.backward(cell.W_ih, TensorEngine.Tensor(∇W_ih; requires_grad=false))
            end
            if cell.W_hh.requires_grad
                TensorEngine.backward(cell.W_hh, TensorEngine.Tensor(∇W_hh; requires_grad=false))
            end
            if input.requires_grad
                TensorEngine.backward(input, TensorEngine.Tensor(∇x; requires_grad=false))
            end
            if hidden.requires_grad
                TensorEngine.backward(hidden, TensorEngine.Tensor(∇h; requires_grad=false))
            end
        end
    end

    return h_new
end

# ------------------------------------------------------------------

mutable struct RNN <: AbstractLayer.Layer
    cell::RNNCell
    batch_first::Bool
    return_sequences::Bool
end

function RNN(input_size::Int, hidden_size::Int;
             batch_first::Bool=true,
             return_sequences::Bool=true,
             bias::Bool=true,
             activation::Function=tanh,
             requires_grad::Bool=true)
    cell = RNNCell(input_size, hidden_size; bias=bias,
                   activation=activation, requires_grad=requires_grad)
    return RNN(cell, batch_first, return_sequences)
end

# =======================
#  RNN FORWARD (arreglado)
# =======================
function forward(rnn::RNN, x::TensorEngine.Tensor,
                 h0::Union{TensorEngine.Tensor, Nothing}=nothing)

    # ---- 1) Normalización de entrada (tu lógica) ----
    if ndims(x.data) == 2
        N, flat_dim = size(x.data)
        if flat_dim == rnn.cell.input_size
            x = TensorEngine.Tensor(reshape(x.data, (N, 1, flat_dim));
                                    requires_grad=x.requires_grad)
        else
            if rnn.batch_first
                T = Int(flat_dim ÷ rnn.cell.input_size)
                D = rnn.cell.input_size
                x = TensorEngine.Tensor(reshape(x.data, (N, T, D));
                                        requires_grad=x.requires_grad)
            else
                T = Int(flat_dim ÷ rnn.cell.input_size)
                D = rnn.cell.input_size
                x = TensorEngine.Tensor(reshape(x.data, (T, N, D));
                                        requires_grad=x.requires_grad)
            end
        end
    end

    # ---- 2) Reordenamiento si batch_first ----
    x_data = rnn.batch_first ? permutedims(x.data, (2, 1, 3)) : x.data  # (T,N,D)
    T, N, D = size(x_data)
    H = rnn.cell.hidden_size
    device = TensorEngine.device_of(x)

    # ---- 3) h0 ----
    if h0 === nothing
        h_data = zeros(Float32, H, N)
        h_data = device == :gpu ? CUDA.cu(h_data) : h_data
        h0 = TensorEngine.Tensor(h_data; requires_grad=false)
    end

    # ---- 4) Loop temporal con PUENTE DE GRADIENTE x_t → x ----
    outputs = TensorEngine.Tensor[]
    hidden  = h0

    for t in 1:T
        # x_data[t, :, :] es (N, D)  → necesitamos (D, N)
        # Usar `view` (función), no `@view` (macro), para evitar el error de precompilación.
        x_slice  = view(x_data, t, :, :)                 # (N, D) en el mismo device
        x_t_data = permutedims(x_slice, (2, 1))          # (D, N)
        x_t = TensorEngine.Tensor(x_t_data; requires_grad=x.requires_grad)

        # *** PUENTE CRÍTICO DE GRADIENTE ***
        if x.requires_grad
            let t_local = t, T_local = T, N_local = N, D_local = D,
                batch_first_local = rnn.batch_first, device_local = device, x_ref = x

                x_t.backward_fn = function(g)
                    gdata = g isa TensorEngine.Tensor ? g.data : g   # (D,N)
                    gNxD  = permutedims(gdata, (2, 1))               # (N,D)

                    # Gradiente con la MISMA forma que x.data
                    if batch_first_local
                        # x: (N,T,D)
                        dX_slice = device_local == :gpu ?
                            CUDA.zeros(Float32, N_local, T_local, D_local) :
                            zeros(Float32, N_local, T_local, D_local)
                        @views dX_slice[:, t_local, :] .= gNxD
                    else
                        # x: (T,N,D)
                        dX_slice = device_local == :gpu ?
                            CUDA.zeros(Float32, T_local, N_local, D_local) :
                            zeros(Float32, T_local, N_local, D_local)
                        @views dX_slice[t_local, :, :] .= gNxD
                    end

                    TensorEngine.backward(x_ref, TensorEngine.Tensor(dX_slice; requires_grad=false))
                end
            end
        end

        hidden = forward(rnn.cell, x_t, hidden)
        push!(outputs, hidden)
    end




    # ---- 5) Salida ----
    if !rnn.return_sequences
        # outputs[end] tiene (H,N). La cadena de RNNCell ya backpropaga a pasos previos.
        return outputs[end]
    end

    # Stack en (T,N,H)
    out_data = zeros(Float32, T, N, H)
    out_data = device == :gpu ? CUDA.cu(out_data) : out_data
    for t in 1:T
        # outputs[t].data: (H,N) -> (N,H)
        @views out_data[t, :, :] .= permutedims(outputs[t].data, (2,1))
    end

    # Ajuste batch_first
    out_data = rnn.batch_first ? permutedims(out_data, (2, 1, 3)) : out_data  # (N,T,H) ó (T,N,H)
    output = TensorEngine.Tensor(out_data; requires_grad=x.requires_grad)

    # ---- 6) Backward de la secuencia completa (para return_sequences=true) ----
    if output.requires_grad
        output.backward_fn = function(grad)
            g = grad isa TensorEngine.Tensor ? grad.data : grad
            g_tnh = rnn.batch_first ? permutedims(g, (2, 1, 3)) : g  # (T,N,H)

            for t in T:-1:1
                # grad_t: (H,N)
                grad_t = permutedims(@view(g_tnh[t, :, :]), (2,1))
                TensorEngine.backward(outputs[t],
                    TensorEngine.Tensor(grad_t; requires_grad=false))
            end
            # Nota: Los backward de cada cell → input activarán los
            # backward_fn de x_t definidos arriba, que a su vez acumulan en x.
        end
    end

    return output
end

# Hacer RNN callable
(rnn::RNN)(input::TensorEngine.Tensor) = forward(rnn, input, nothing)
# ==================================================================
# Layers.jl - LSTM/GRU con backward correctamente conectado
# ==================================================================

# Helper crítico que faltaba
function _ensure_tensor(x, requires_grad::Bool)
    if x isa TensorEngine.Tensor
        return x
    else
        return TensorEngine.Tensor(x; requires_grad=requires_grad)
    end
end

# ==================================================================
# Utilidad: normaliza entrada a (F, B, T)
# Acepta 3D (F,B,T) o 4D (F,B,1,T) / (F,B,T,1) y reduce dimensión unitaria
# ==================================================================
@inline function _to3d_sequence(data)
    nd = ndims(data)
    if nd == 3
        return data
    elseif nd == 2
        F, B = size(data)
        return data isa CUDA.CuArray ? CUDA.reshape(data, (F, B, 1)) :
                                       reshape(data, (F, B, 1))
    elseif nd == 4
        F, B, D3, D4 = size(data)
        if D3 == 1
            return data isa CUDA.CuArray ? CUDA.reshape(data, (F, B, D4)) :
                                           reshape(data, (F, B, D4))
        elseif D4 == 1
            return data isa CUDA.CuArray ? CUDA.reshape(data, (F, B, D3)) :
                                           reshape(data, (F, B, D3))
        else
            throw(ArgumentError("Recurrent layers expect (F,B,T) or 4D with a singleton time/channel dim; got size=$(size(data))"))
        end
    else
        throw(ArgumentError("Recurrent layers expect 2D, 3D or 4D input; got ndims=$(nd)"))
    end
end

# ==================================================================
# Activaciones que preservan gradiente (sin helpers externos)
# ==================================================================
function sigmoid_tensor(x::TensorEngine.Tensor)
    # Sigmoide estable numéricamente (evita overflow para |x| grande)
    y_data = 1f0 ./ (1f0 .+ exp.(-x.data))
    out = TensorEngine.Tensor(y_data; requires_grad=x.requires_grad)

    if x.requires_grad
        out.backward_fn = function(grad)
            grad_data = grad isa TensorEngine.Tensor ? grad.data : grad
            dx = grad_data .* y_data .* (1f0 .- y_data)
            TensorEngine.backward(x, TensorEngine.Tensor(dx))
        end
    end

    return out
end

function tanh_tensor(x::TensorEngine.Tensor)
    y = tanh.(x.data)
    out = TensorEngine.Tensor(y; requires_grad=x.requires_grad)

    if x.requires_grad
        out.backward_fn = function(grad)
            grad_data = grad isa TensorEngine.Tensor ? grad.data : grad
            dx = grad_data .* (1f0 .- y.^2)
            TensorEngine.backward(x, TensorEngine.Tensor(dx))
        end
    end

    return out
end


# ==================================================================
# Estructuras
# ==================================================================
mutable struct LSTMCell <: AbstractLayer.Layer
    input_size::Int
    hidden_size::Int
    W_ii::TensorEngine.Tensor; W_if::TensorEngine.Tensor
    W_ig::TensorEngine.Tensor; W_io::TensorEngine.Tensor
    W_hi::TensorEngine.Tensor; W_hf::TensorEngine.Tensor
    W_hg::TensorEngine.Tensor; W_ho::TensorEngine.Tensor
    b_ii::TensorEngine.Tensor; b_if::TensorEngine.Tensor
    b_ig::TensorEngine.Tensor; b_io::TensorEngine.Tensor
    b_hi::TensorEngine.Tensor; b_hf::TensorEngine.Tensor
    b_hg::TensorEngine.Tensor; b_ho::TensorEngine.Tensor
    training::Bool
end

mutable struct GRUCell <: AbstractLayer.Layer
    input_size::Int
    hidden_size::Int
    W_ir::TensorEngine.Tensor; W_iz::TensorEngine.Tensor; W_in::TensorEngine.Tensor
    W_hr::TensorEngine.Tensor; W_hz::TensorEngine.Tensor; W_hn::TensorEngine.Tensor
    b_ir::TensorEngine.Tensor; b_iz::TensorEngine.Tensor; b_in::TensorEngine.Tensor
    b_hr::TensorEngine.Tensor; b_hz::TensorEngine.Tensor; b_hn::TensorEngine.Tensor
    training::Bool
end

# ==================================================================
# Constructores
# ==================================================================
function LSTMCell(input_size::Int, hidden_size::Int; 
                  bias::Bool=true, requires_grad::Bool=true)
    σ = sqrt(2f0 / (input_size + hidden_size))
    
    W_ii = TensorEngine.Tensor(randn(Float32, hidden_size, input_size) .* σ; requires_grad)
    W_if = TensorEngine.Tensor(randn(Float32, hidden_size, input_size) .* σ; requires_grad)
    W_ig = TensorEngine.Tensor(randn(Float32, hidden_size, input_size) .* σ; requires_grad)
    W_io = TensorEngine.Tensor(randn(Float32, hidden_size, input_size) .* σ; requires_grad)
    
    W_hi = TensorEngine.Tensor(randn(Float32, hidden_size, hidden_size) .* σ; requires_grad)
    W_hf = TensorEngine.Tensor(randn(Float32, hidden_size, hidden_size) .* σ; requires_grad)
    W_hg = TensorEngine.Tensor(randn(Float32, hidden_size, hidden_size) .* σ; requires_grad)
    W_ho = TensorEngine.Tensor(randn(Float32, hidden_size, hidden_size) .* σ; requires_grad)
    
    b_ii = TensorEngine.Tensor(zeros(Float32, hidden_size, 1); requires_grad)
    b_if = TensorEngine.Tensor(ones(Float32, hidden_size, 1); requires_grad)  # Forget bias = 1
    b_ig = TensorEngine.Tensor(zeros(Float32, hidden_size, 1); requires_grad)
    b_io = TensorEngine.Tensor(zeros(Float32, hidden_size, 1); requires_grad)
    
    b_hi = TensorEngine.Tensor(zeros(Float32, hidden_size, 1); requires_grad)
    b_hf = TensorEngine.Tensor(zeros(Float32, hidden_size, 1); requires_grad)
    b_hg = TensorEngine.Tensor(zeros(Float32, hidden_size, 1); requires_grad)
    b_ho = TensorEngine.Tensor(zeros(Float32, hidden_size, 1); requires_grad)
    
    return LSTMCell(input_size, hidden_size,
                    W_ii, W_if, W_ig, W_io,
                    W_hi, W_hf, W_hg, W_ho,
                    b_ii, b_if, b_ig, b_io,
                    b_hi, b_hf, b_hg, b_ho, true)
end

function GRUCell(input_size::Int, hidden_size::Int; 
                 bias::Bool=true, requires_grad::Bool=true)
    σ = sqrt(2f0 / (input_size + hidden_size))
    
    W_ir = TensorEngine.Tensor(randn(Float32, hidden_size, input_size) .* σ; requires_grad)
    W_iz = TensorEngine.Tensor(randn(Float32, hidden_size, input_size) .* σ; requires_grad)
    W_in = TensorEngine.Tensor(randn(Float32, hidden_size, input_size) .* σ; requires_grad)
    
    W_hr = TensorEngine.Tensor(randn(Float32, hidden_size, hidden_size) .* σ; requires_grad)
    W_hz = TensorEngine.Tensor(randn(Float32, hidden_size, hidden_size) .* σ; requires_grad)
    W_hn = TensorEngine.Tensor(randn(Float32, hidden_size, hidden_size) .* σ; requires_grad)
    
    b_ir = TensorEngine.Tensor(zeros(Float32, hidden_size, 1); requires_grad)
    b_iz = TensorEngine.Tensor(zeros(Float32, hidden_size, 1); requires_grad)
    b_in = TensorEngine.Tensor(zeros(Float32, hidden_size, 1); requires_grad)
    
    b_hr = TensorEngine.Tensor(zeros(Float32, hidden_size, 1); requires_grad)
    b_hz = TensorEngine.Tensor(zeros(Float32, hidden_size, 1); requires_grad)
    b_hn = TensorEngine.Tensor(zeros(Float32, hidden_size, 1); requires_grad)
    
    return GRUCell(input_size, hidden_size,
                   W_ir, W_iz, W_in,
                   W_hr, W_hz, W_hn,
                   b_ir, b_iz, b_in,
                   b_hr, b_hz, b_hn, true)
end

# ==================================================================
# Forward LSTM con backward conectado
# ==================================================================
function forward(cell::LSTMCell, input::TensorEngine.Tensor, state)
    # Normalizar estados
    h_prev = state[1] isa TensorEngine.Tensor ? state[1] : TensorEngine.Tensor(state[1]; requires_grad=input.requires_grad)
    c_prev = state[2] isa TensorEngine.Tensor ? state[2] : TensorEngine.Tensor(state[2]; requires_grad=input.requires_grad)
    
    # Mover todo al mismo dispositivo
    device = TensorEngine.device_of(input)
    h_prev = TensorEngine.ensure_on_device(h_prev, device)
    c_prev = TensorEngine.ensure_on_device(c_prev, device)
    
    # Usar TensorEngine.matmul que devuelve Tensor
    mm = TensorEngine.matmul
    
    # Computar pre-activaciones como Tensors
    zi = mm(cell.W_ii, input)
    zf = mm(cell.W_if, input)  
    zg = mm(cell.W_ig, input)
    zo = mm(cell.W_io, input)
    
    ri = mm(cell.W_hi, h_prev)
    rf = mm(cell.W_hf, h_prev)
    rg = mm(cell.W_hg, h_prev)
    ro = mm(cell.W_ho, h_prev)
    
    # Sumar biases (operaciones Tensor + Tensor)
    pre_i = TensorEngine.add(TensorEngine.add(zi, cell.b_ii), TensorEngine.add(ri, cell.b_hi))
    pre_f = TensorEngine.add(TensorEngine.add(zf, cell.b_if), TensorEngine.add(rf, cell.b_hf))
    pre_g = TensorEngine.add(TensorEngine.add(zg, cell.b_ig), TensorEngine.add(rg, cell.b_hg))
    pre_o = TensorEngine.add(TensorEngine.add(zo, cell.b_io), TensorEngine.add(ro, cell.b_ho))
    
    # Aplicar activaciones
    i_gate = sigmoid_tensor(pre_i)
    f_gate = sigmoid_tensor(pre_f)
    g_gate = tanh_tensor(pre_g)
    o_gate = sigmoid_tensor(pre_o)
    
    # Cell state update (elemento a elemento entre Tensors)
    c_new_part1 = TensorEngine.Tensor(f_gate.data .* c_prev.data; requires_grad=input.requires_grad)
    c_new_part2 = TensorEngine.Tensor(i_gate.data .* g_gate.data; requires_grad=input.requires_grad)
    c_new = TensorEngine.add(c_new_part1, c_new_part2)
    
    # Clamp para estabilidad
    c_new = TensorEngine.Tensor(clamp.(c_new.data, -10f0, 10f0); requires_grad=input.requires_grad)
    
    # Hidden state
    tanh_c = tanh_tensor(c_new)
    h_new = TensorEngine.Tensor(o_gate.data .* tanh_c.data; requires_grad=input.requires_grad)
    
    # CONECTAR BACKWARD
    if h_new.requires_grad
        # Guardar valores para backward
        saved_for_backward = (i_gate.data, f_gate.data, g_gate.data, o_gate.data, 
                              c_prev.data, c_new.data, h_prev.data, input.data)
        
        h_new.backward_fn = function(grad_h)
            grad_h_data = grad_h isa TensorEngine.Tensor ? grad_h.data : grad_h
            
            # Desempaquetar
            i_g, f_g, g_g, o_g, c_old, c_n, h_old, x = saved_for_backward
            
            # Backward through output
            grad_o = grad_h_data .* tanh.(c_n)
            grad_c_from_h = grad_h_data .* o_g .* (1f0 .- tanh.(c_n).^2)
            
            # Backward through cell
            grad_f = grad_c_from_h .* c_old
            grad_i = grad_c_from_h .* g_g
            grad_g = grad_c_from_h .* i_g
            grad_c_prev = grad_c_from_h .* f_g
            
            # Through gates
            grad_i_pre = grad_i .* i_g .* (1f0 .- i_g)
            grad_f_pre = grad_f .* f_g .* (1f0 .- f_g)
            grad_g_pre = grad_g .* (1f0 .- g_g.^2)
            grad_o_pre = grad_o .* o_g .* (1f0 .- o_g)
            
            # Weight gradients
            TensorEngine.backward(cell.W_ii, TensorEngine.Tensor(grad_i_pre * x'))
            TensorEngine.backward(cell.W_if, TensorEngine.Tensor(grad_f_pre * x'))
            TensorEngine.backward(cell.W_ig, TensorEngine.Tensor(grad_g_pre * x'))
            TensorEngine.backward(cell.W_io, TensorEngine.Tensor(grad_o_pre * x'))
            
            TensorEngine.backward(cell.W_hi, TensorEngine.Tensor(grad_i_pre * h_old'))
            TensorEngine.backward(cell.W_hf, TensorEngine.Tensor(grad_f_pre * h_old'))
            TensorEngine.backward(cell.W_hg, TensorEngine.Tensor(grad_g_pre * h_old'))
            TensorEngine.backward(cell.W_ho, TensorEngine.Tensor(grad_o_pre * h_old'))
            
            # Bias gradients
            TensorEngine.backward(cell.b_ii, TensorEngine.Tensor(sum(grad_i_pre, dims=2)))
            TensorEngine.backward(cell.b_if, TensorEngine.Tensor(sum(grad_f_pre, dims=2)))
            TensorEngine.backward(cell.b_ig, TensorEngine.Tensor(sum(grad_g_pre, dims=2)))
            TensorEngine.backward(cell.b_io, TensorEngine.Tensor(sum(grad_o_pre, dims=2)))
            
            TensorEngine.backward(cell.b_hi, TensorEngine.Tensor(sum(grad_i_pre, dims=2)))
            TensorEngine.backward(cell.b_hf, TensorEngine.Tensor(sum(grad_f_pre, dims=2)))
            TensorEngine.backward(cell.b_hg, TensorEngine.Tensor(sum(grad_g_pre, dims=2)))
            TensorEngine.backward(cell.b_ho, TensorEngine.Tensor(sum(grad_o_pre, dims=2)))
            
            # Input gradient  
            grad_x = cell.W_ii.data' * grad_i_pre .+ 
                     cell.W_if.data' * grad_f_pre .+ 
                     cell.W_ig.data' * grad_g_pre .+ 
                     cell.W_io.data' * grad_o_pre
            TensorEngine.backward(input, TensorEngine.Tensor(grad_x))
            
            # Hidden gradient
            grad_h_prev = cell.W_hi.data' * grad_i_pre .+ 
                          cell.W_hf.data' * grad_f_pre .+ 
                          cell.W_hg.data' * grad_g_pre .+ 
                          cell.W_ho.data' * grad_o_pre
            TensorEngine.backward(h_prev, TensorEngine.Tensor(grad_h_prev))
            TensorEngine.backward(c_prev, TensorEngine.Tensor(grad_c_prev))
        end
        
        # También backward para c_new
        c_new.backward_fn = function(grad_c)
            # Propagar gradiente del cell state
            if h_new.backward_fn !== nothing
                # Esto se maneja arriba
            end
        end
    end
    
    return (h_new, c_new)
end

(cell::LSTMCell)(input, state) = forward(cell, input, state)

# ==================================================================
# Forward GRU con backward conectado
# ==================================================================
function forward(cell::GRUCell, input::TensorEngine.Tensor, hidden)
    h_prev = hidden isa TensorEngine.Tensor ? hidden : TensorEngine.Tensor(hidden; requires_grad=input.requires_grad)
    
    device = TensorEngine.device_of(input)
    h_prev = TensorEngine.ensure_on_device(h_prev, device)
    
    mm = TensorEngine.matmul
    
    # Pre-activaciones
    zi = mm(cell.W_ir, input)
    zz = mm(cell.W_iz, input)
    zn = mm(cell.W_in, input)
    
    ri = mm(cell.W_hr, h_prev)
    rz = mm(cell.W_hz, h_prev)
    rn = mm(cell.W_hn, h_prev)
    
    pre_r = TensorEngine.add(TensorEngine.add(zi, cell.b_ir), TensorEngine.add(ri, cell.b_hr))
    pre_z = TensorEngine.add(TensorEngine.add(zz, cell.b_iz), TensorEngine.add(rz, cell.b_hz))
    
    r_gate = sigmoid_tensor(pre_r)
    z_gate = sigmoid_tensor(pre_z)
    
    # New gate con reset aplicado
    rn_gated = TensorEngine.Tensor(r_gate.data .* rn.data; requires_grad=input.requires_grad)
    pre_n = TensorEngine.add(TensorEngine.add(zn, cell.b_in), 
                             TensorEngine.add(rn_gated, cell.b_hn))
    n_gate = tanh_tensor(pre_n)
    
    # Hidden update
    h_new_part1 = TensorEngine.Tensor((1f0 .- z_gate.data) .* n_gate.data; requires_grad=input.requires_grad)
    h_new_part2 = TensorEngine.Tensor(z_gate.data .* h_prev.data; requires_grad=input.requires_grad)
    h_new = TensorEngine.add(h_new_part1, h_new_part2)
    
    # CONECTAR BACKWARD
    if h_new.requires_grad
        saved = (r_gate.data, z_gate.data, n_gate.data, h_prev.data, input.data)
        
        h_new.backward_fn = function(grad_h)
            grad_h_data = grad_h isa TensorEngine.Tensor ? grad_h.data : grad_h
            r_g, z_g, n_g, h_old, x = saved

            grad_n = grad_h_data .* (1f0 .- z_g)
            grad_z = grad_h_data .* (n_g .- h_old)
            grad_h_old_direct = grad_h_data .* z_g

            grad_n_pre = grad_n .* (1f0 .- n_g.^2)
            grad_z_pre = grad_z .* z_g .* (1f0 .- z_g)

            Whn_h = cell.W_hn.data * h_old
            grad_r = grad_n_pre .* Whn_h
            grad_r_pre = grad_r .* r_g .* (1f0 .- r_g)

            TensorEngine.backward(cell.W_ir, TensorEngine.Tensor(grad_r_pre * x'))
            TensorEngine.backward(cell.W_iz, TensorEngine.Tensor(grad_z_pre * x'))
            TensorEngine.backward(cell.W_in, TensorEngine.Tensor(grad_n_pre * x'))

            TensorEngine.backward(cell.W_hr, TensorEngine.Tensor(grad_r_pre * h_old'))
            TensorEngine.backward(cell.W_hz, TensorEngine.Tensor(grad_z_pre * h_old'))
            TensorEngine.backward(cell.W_hn, TensorEngine.Tensor(grad_n_pre .* r_g * h_old'))

            TensorEngine.backward(cell.b_ir, TensorEngine.Tensor(sum(grad_r_pre, dims=2)))
            TensorEngine.backward(cell.b_iz, TensorEngine.Tensor(sum(grad_z_pre, dims=2)))
            TensorEngine.backward(cell.b_in, TensorEngine.Tensor(sum(grad_n_pre, dims=2)))

            TensorEngine.backward(cell.b_hr, TensorEngine.Tensor(sum(grad_r_pre, dims=2)))
            TensorEngine.backward(cell.b_hz, TensorEngine.Tensor(sum(grad_z_pre, dims=2)))
            TensorEngine.backward(cell.b_hn, TensorEngine.Tensor(sum(grad_n_pre, dims=2)))  # ← fix

            grad_x = cell.W_ir.data' * grad_r_pre .+
                    cell.W_iz.data' * grad_z_pre .+
                    cell.W_in.data' * grad_n_pre
            TensorEngine.backward(input, TensorEngine.Tensor(grad_x))

            grad_h_prev = grad_h_old_direct .+
                        cell.W_hr.data' * grad_r_pre .+
                        cell.W_hz.data' * grad_z_pre .+
                        cell.W_hn.data' * (grad_n_pre .* r_g)
            TensorEngine.backward(h_prev, TensorEngine.Tensor(grad_h_prev))
        end

    end
    
    return h_new
end

(cell::GRUCell)(input, hidden) = forward(cell, input, hidden)

# ==================================================================
# Wrappers LSTM/GRU (más limpios)
# ==================================================================
mutable struct LSTM <: AbstractLayer.Layer
    cell::LSTMCell
    return_sequences::Bool
end

function LSTM(input_size::Int, hidden_size::Int; 
              return_sequences::Bool=false, kwargs...)
    cell = LSTMCell(input_size, hidden_size; kwargs...)
    return LSTM(cell, return_sequences)
end

function forward(lstm::LSTM, input::TensorEngine.Tensor, state=nothing)
    data3 = _to3d_sequence(input.data)

    input3 =
        (data3 === input.data) ? input :
        begin
            t3 = TensorEngine.Tensor(data3; requires_grad=input.requires_grad)
            if t3.requires_grad
                # puente backward 3D -> input original (2D/3D/4D)
                t3.backward_fn = function(grad3)
                    g3 = grad3 isa TensorEngine.Tensor ? grad3.data : grad3
                    if ndims(input.data) == 3
                        TensorEngine.backward(input, TensorEngine.Tensor(g3))
                    else
                        F, B = size(g3, 1), size(g3, 2)
                        if size(input.data, 3) == 1
                            g4 = (input.data isa CUDA.CuArray) ?
                                 CUDA.reshape(g3, (F, B, 1, size(g3,3))) :
                                 reshape(g3, (F, B, 1, size(g3,3)))
                        else
                            g4 = (input.data isa CUDA.CuArray) ?
                                 CUDA.reshape(g3, (F, B, size(g3,3), 1)) :
                                 reshape(g3, (F, B, size(g3,3), 1))
                        end
                        TensorEngine.backward(input, TensorEngine.Tensor(g4))
                    end
                end
            end
            t3
        end

    seq_len = size(data3, 3)
    batch_size = size(data3, 2)
    device = TensorEngine.device_of(input3)

    if state === nothing
        h = zeros(Float32, lstm.cell.hidden_size, batch_size)
        c = zeros(Float32, lstm.cell.hidden_size, batch_size)
        if device == :gpu
            h = CUDA.CuArray(h); c = CUDA.CuArray(c)
        end
        state = (TensorEngine.Tensor(h), TensorEngine.Tensor(c))
    end

    outputs = Vector{TensorEngine.Tensor}(undef, seq_len)
    for t in 1:seq_len
        x_t_data = data3[:, :, t]
        x_t = TensorEngine.Tensor(copy(x_t_data); requires_grad=input3.requires_grad)

        if x_t.requires_grad
            # puente backward timestep -> gradiente 3D completo hacia input3
            x_t.backward_fn = function(grad_xt)
                g = grad_xt isa TensorEngine.Tensor ? grad_xt.data : grad_xt
                G = similar(data3, size(data3))
                fill!(G, zero(eltype(G)))
                @views G[:, :, t] .= g
                TensorEngine.backward(input3, TensorEngine.Tensor(G))
            end
        end

        state = forward(lstm.cell, x_t, state)
        outputs[t] = state[1]
    end

    if lstm.return_sequences
        out_data = cat((o.data for o in outputs)..., dims=3)
        return TensorEngine.Tensor(out_data; requires_grad=input3.requires_grad)
    else
        return outputs[end]
    end
end


(lstm::LSTM)(input, state=nothing) = forward(lstm, input, state)

mutable struct GRU <: AbstractLayer.Layer
    cell::GRUCell
    return_sequences::Bool
end

function GRU(input_size::Int, hidden_size::Int; 
             return_sequences::Bool=false, kwargs...)
    cell = GRUCell(input_size, hidden_size; kwargs...)
    return GRU(cell, return_sequences)
end

function forward(gru::GRU, input::TensorEngine.Tensor, hidden=nothing)
    # Normalizar a (F, B, T)
    data3 = _to3d_sequence(input.data)
    input3 = (data3 === input.data) ? input :
             TensorEngine.Tensor(data3; requires_grad=input.requires_grad)

    seq_len   = size(data3, 3)
    batch_size = size(data3, 2)
    device = TensorEngine.device_of(input3)

    if hidden === nothing
        h = zeros(Float32, gru.cell.hidden_size, batch_size)
        if device == :gpu
            h = CUDA.CuArray(h)
        end
        hidden = TensorEngine.Tensor(h)
    end

    outputs = Vector{TensorEngine.Tensor}(undef, seq_len)
    for t in 1:seq_len
        x_t_data = data3[:, :, t]
        x_t = TensorEngine.Tensor(copy(x_t_data); requires_grad=input3.requires_grad)
        hidden = forward(gru.cell, x_t, hidden)
        outputs[t] = hidden
    end

    if gru.return_sequences
        out_data = cat((o.data for o in outputs)..., dims=3)
        return TensorEngine.Tensor(out_data; requires_grad=input3.requires_grad)
    else
        return outputs[end]
    end
end

(gru::GRU)(input, hidden=nothing) = forward(gru, input, hidden)


end  # module Layers