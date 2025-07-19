module Layers

using NNlib, CUDA, ..TensorEngine, ..AbstractLayer, ..ConvKernelLayers, Statistics
export BatchNorm, Flatten, LayerActivation, GlobalAvgPool, ResidualBlock, create_residual_block, CustomFlatten, DropoutLayer
# Añadir en Layers.jl justo antes del end del módulo

# ==================================================================
# BatchNorm Optimizado (Fusión GPU)
# ==================================================================
"""
Versión corregida de BatchNormalization que usa Float32 explícitamente
"""
mutable struct BatchNorm <: AbstractLayer.Layer
    gamma::TensorEngine.Tensor
    beta::TensorEngine.Tensor
    running_mean::Array{Float32}
    running_var::Array{Float32}
    momentum::Float32
    epsilon::Float32
    training::Bool
end

function BatchNorm(channels::Int; momentum=Float32(0.9), epsilon=Float32(1e-5), training=true)
    # Inicializar todos los parámetros con tipos Float32 explícitos
    gamma = TensorEngine.Tensor(ones(Float32, channels))
    beta = TensorEngine.Tensor(zeros(Float32, channels))
    running_mean = zeros(Float32, channels)
    running_var = ones(Float32, channels)
    
    return BatchNorm(gamma, beta, running_mean, running_var, momentum, epsilon, training)
end

function forward(layer::BatchNorm, input::TensorEngine.Tensor)
    x = input.data
    # Asegurarse que los datos sean de tipo Float32
    if eltype(x) != Float32
        x = Float32.(x)
    end
    
    # Determinar dimensiones
    dims = ndims(x)
    batch_size = size(x, 1)
    num_channels = size(x, 2)
    spatial_dims = size(x)[3:end]
    
    # Imprimir información de depuración
    #println("BatchNorm: input shape = $(size(x)), gamma shape = $(size(layer.gamma.data))")
    
    if layer.training
        # Normalizar por batch y dimensiones espaciales, pero no por canal
        # Para tensores 4D: (batch, channel, height, width)
        reduce_dims = (1, 3, 4)  # Normalizar por batch y dimensiones espaciales
        
        # Calcular media y varianza por canal
        batch_mean = dropdims(mean(x, dims=reduce_dims), dims=reduce_dims)
        batch_var = dropdims(var(x, dims=reduce_dims), dims=reduce_dims)
        
        # Asegurarnos que tienen dimensiones correctas para broadcast
        # Para 4D: (1, C, 1, 1)
        reshape_dims = (1, num_channels, ntuple(i -> 1, dims-2)...)
        
        reshaped_mean = reshape(batch_mean, reshape_dims)
        reshaped_var = reshape(batch_var, reshape_dims)
        
        # Actualizar running stats
        layer.running_mean = layer.momentum .* layer.running_mean .+ (1 - layer.momentum) .* Array(batch_mean)
        layer.running_var = layer.momentum .* layer.running_var .+ (1 - layer.momentum) .* Array(batch_var)
        
        # Normalizar
        x_normalized = (x .- reshaped_mean) ./ sqrt.(reshaped_var .+ layer.epsilon)
    else
        # Preparar running stats para broadcast
        reshape_dims = (1, num_channels)
        for _ in 1:dims-2
            push!(reshape_dims, 1)
        end
        
        reshaped_mean = reshape(layer.running_mean, reshape_dims)
        reshaped_var = reshape(layer.running_var, reshape_dims)
        
        # Normalizar usando estadísticas acumuladas
        x_normalized = (x .- reshaped_mean) ./ sqrt.(reshaped_var .+ layer.epsilon)
    end
    
    # Preparar gamma y beta para broadcast
    gamma_shape = reshape_dims
    beta_shape = reshape_dims
    
    # Asegurarse de que gamma y beta tengan dimensiones correctas
    gamma_reshaped = reshape(layer.gamma.data, gamma_shape)
    beta_reshaped = reshape(layer.beta.data, beta_shape)
    
    # Aplicar parámetros gamma y beta
    output = gamma_reshaped .* x_normalized .+ beta_reshaped
    
    # Crear tensor de salida 
    out = TensorEngine.Tensor(output)
    
    return out
end


# Hacer que la capa sea callable directamente
function (layer::BatchNorm)(input::TensorEngine.Tensor)
    return forward(layer, input)
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