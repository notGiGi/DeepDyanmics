module NeuralNetwork

using ..TensorEngine
using ..Layers
using ..ConvolutionalLayers
using ..EmbeddingLayer
using ..AbstractLayer
import NNlib: conv
using CUDA.CUBLAS: gemm!
using CUDA
using LinearAlgebra
using ..ReshapeModule: Reshape  # Importar Reshape desde ReshapeModule
using ..ConvKernelLayers
export Sequential, Dense, collect_parameters, relu, sigmoid, tanh_activation,
       leaky_relu, swish, mish, softmax, Activation

# ==================================================================
# Capa Dense Optimizada (Sin activación integrada)
# ==================================================================
mutable struct Dense <: AbstractLayer.Layer
    weights::TensorEngine.Tensor
    biases::TensorEngine.Tensor
end

function Dense(input_size::Int, output_size::Int; init_method::Symbol=:xavier)
    weights = TensorEngine.initialize_weights((input_size, output_size); method=init_method)
    bias_data = CUDA.functional() ? CUDA.zeros(Float32, output_size, 1) : zeros(Float32, output_size, 1)
    biases = TensorEngine.Tensor(0.01f0 .* bias_data)
    Dense(weights, biases)
end

function forward(layer::Dense, input::TensorEngine.Tensor)
    # Verificar dimensiones de entrada
    expected_input_size = size(layer.weights.data, 2)
    actual_input_size = size(input.data, 1)
    
    @assert actual_input_size == expected_input_size "Dense layer expects input size $expected_input_size but got $actual_input_size"
    
    # Obtener datos
    W = layer.weights.data
    x = input.data
    
    # Detectar si estamos trabajando en GPU o CPU
    w_on_gpu = (W isa CUDA.CuArray)
    x_on_gpu = (x isa CUDA.CuArray)
    
    # Asegurar que W y x estén en el mismo dispositivo
    if w_on_gpu && !x_on_gpu
        x = CUDA.CuArray(x)
        x_on_gpu = true
    elseif !w_on_gpu && x_on_gpu
        W = CUDA.CuArray(W)
        w_on_gpu = true
    end
    
    # Realizar la multiplicación de matrices
    output = W * x
    
    # Obtener el bias y asegurar que esté en el mismo dispositivo
    b = layer.biases.data
    b_on_gpu = (b isa CUDA.CuArray)
    
    if b_on_gpu != x_on_gpu
        if x_on_gpu
            b = CUDA.CuArray(b)
        else
            b = Array(b)
        end
    end
    
    # Añadir el bias
    # reshape para asegurar broadcasteo correcto
    output = output .+ reshape(b, :, 1)
    
    # Crear tensor de salida
    out_tensor = TensorEngine.Tensor(output)
    
    # Definir la función backward
    out_tensor.backward_fn = function(grad)
        # Asegurar que el gradiente está en el mismo dispositivo
        grad_on_gpu = (grad isa CUDA.CuArray)
        grad_data = grad
        
        if grad_on_gpu != x_on_gpu
            if x_on_gpu
                grad_data = CUDA.CuArray(grad_data)
            else
                grad_data = Array(grad_data)
            end
        end
        
        # Calcular gradientes
        ∇W = grad_data * x'
        ∇b = sum(grad_data, dims=2)
        ∇x = W' * grad_data
        
        # Propagar gradientes
        TensorEngine.backward(layer.weights, TensorEngine.Tensor(∇W))
        TensorEngine.backward(layer.biases, TensorEngine.Tensor(∇b))
        TensorEngine.backward(input, TensorEngine.Tensor(∇x))
    end
    
    return out_tensor
end

function (layer::Dense)(input::TensorEngine.Tensor)
    return forward(layer, input)
end

# ==================================================================
# Capa Sequential
# ==================================================================
struct Sequential <: AbstractLayer.Layer
    layers::Vector{AbstractLayer.Layer}
end

function forward(model::Sequential, input::TensorEngine.Tensor; verbose::Bool=false)
    current = input
    for (i, layer) in enumerate(model.layers)
        current = layer(current)
        verbose && @info "Layer $i output size: $(size(current.data))"
    end
    return current
end

function (model::Sequential)(input::TensorEngine.Tensor)
    return forward(model, input)
end

# ==================================================================
# Capa de Activación (corregida)
# ==================================================================
struct Activation <: AbstractLayer.Layer
    f::Function
end

# Método de forward para Activation (cuando se recibe un Tensor)
function (a::Activation)(input::TensorEngine.Tensor)
    return a.f(input)
end

# Nuevo: Método de forward para Activation que acepta un Array y lo convierte a Tensor
function (a::Activation)(x::AbstractArray)
    return a(TensorEngine.Tensor(x))
end

# ==================================================================
# Funciones de Activación Optimizadas
# ==================================================================
function relu(t::TensorEngine.Tensor)
    data_out = max.(t.data, 0.0f0)
    out = TensorEngine.Tensor(data_out)
    out.backward_fn = grad -> begin
        TensorEngine.backward(t, TensorEngine.Tensor((t.data .> 0) .* grad))
    end
    return out
end

function sigmoid(t::TensorEngine.Tensor)
    data_out = 1.0f0 ./ (1.0f0 .+ exp.(-t.data))
    out = TensorEngine.Tensor(data_out)
    out.backward_fn = grad -> begin
        TensorEngine.backward(t, TensorEngine.Tensor(data_out .* (1.0f0 .- data_out) .* grad))
    end
    return out
end

function tanh_activation(t::TensorEngine.Tensor)
    data_out = tanh.(t.data)
    out = TensorEngine.Tensor(data_out)
    out.backward_fn = grad -> begin
        TensorEngine.backward(t, TensorEngine.Tensor((1.0f0 .- data_out.^2) .* grad))
    end
    return out
end

function leaky_relu(t::TensorEngine.Tensor; α=0.01f0)
    data_out = max.(t.data, α .* t.data)
    out = TensorEngine.Tensor(data_out)
    out.backward_fn = grad -> begin
        derivative = map(x -> x > 0 ? 1.0f0 : α, t.data)
        TensorEngine.backward(t, TensorEngine.Tensor(derivative .* grad))
    end
    return out
end

function swish(t::TensorEngine.Tensor)
    s = 1.0f0 ./ (1.0f0 .+ exp.(-t.data))
    out_data = t.data .* s
    out = TensorEngine.Tensor(out_data)
    out.backward_fn = grad -> begin
        grad_val = s .+ t.data .* s .* (1.0f0 .- s)
        TensorEngine.backward(t, TensorEngine.Tensor(grad .* grad_val))
    end
    return out
end

function mish(t::TensorEngine.Tensor)
    softplus = log.(1.0f0 .+ exp.(t.data))
    tanh_sp = tanh.(softplus)
    out_data = t.data .* tanh_sp
    out = TensorEngine.Tensor(out_data)
    out.backward_fn = grad -> begin
        δ = tanh_sp .+ t.data .* (1.0f0 .- tanh_sp.^2) .* (exp.(t.data) ./ (1.0f0 .+ exp.(t.data)))
        TensorEngine.backward(t, TensorEngine.Tensor(grad .* δ))
    end
    return out
end

function softmax(t::TensorEngine.Tensor)::Tensor
    max_val = maximum(t.data)
    log_sum_exp = log.(sum(exp.(t.data .- max_val))) + max_val
    probs = exp.(t.data .- log_sum_exp)
    out = TensorEngine.Tensor(probs)
    out.backward_fn = grad -> begin
        grad_input = probs .* (grad .- sum(probs .* grad, dims=1))
        TensorEngine.backward(t, TensorEngine.Tensor(grad_input))
    end
    return out
end

# ==================================================================
# Funciones de Ayuda
# ==================================================================
function collect_parameters(model::Sequential)
    params = TensorEngine.Tensor[]
    for layer in model.layers
        if layer isa Dense
            push!(params, layer.weights, layer.biases)
        elseif layer isa Layers.BatchNorm  # Usar namespace completo
            push!(params, layer.gamma, layer.beta)
        elseif layer isa ConvKernelLayers.ConvKernelLayer
            # Verificar si ya son Tensors antes de convertir
            if layer.weights isa TensorEngine.Tensor
                push!(params, layer.weights)
            else
                push!(params, TensorEngine.Tensor(layer.weights))
            end
            if layer.bias isa TensorEngine.Tensor
                push!(params, layer.bias)
            else
                push!(params, TensorEngine.Tensor(layer.bias))
            end
        elseif layer isa ConvolutionalLayers.Conv2D
            push!(params, layer.weights, layer.bias)
            if layer.use_batchnorm && layer.gamma !== nothing
                push!(params, layer.gamma, layer.beta)
            end
        elseif layer isa Layers.ResidualBlock
            # Recolectar parámetros de bloques residuales recursivamente
            for sublayer in layer.conv_path
                append!(params, collect_parameters(Sequential([sublayer])))
            end
            for sublayer in layer.shortcut
                append!(params, collect_parameters(Sequential([sublayer])))
            end
        end
        # Activation, Flatten, MaxPooling, etc. no tienen parámetros
    end
    return params
end

function model_to_gpu(model::Sequential)
    layers_gpu = []
    for layer in model.layers
        if layer isa Dense
            weights_gpu = TensorEngine.Tensor(CUDA.CuArray(layer.weights.data))
            biases_gpu  = TensorEngine.Tensor(CUDA.CuArray(layer.biases.data))
            push!(layers_gpu, Dense(weights_gpu, biases_gpu))
        elseif layer isa ConvolutionalLayers.Conv2D
            weights_gpu = TensorEngine.Tensor(CUDA.CuArray(layer.weights.data))
            bias_gpu    = TensorEngine.Tensor(CUDA.CuArray(layer.bias.data))
            gamma_gpu   = layer.use_batchnorm ? TensorEngine.Tensor(CUDA.CuArray(layer.gamma.data)) : nothing
            beta_gpu    = layer.use_batchnorm ? TensorEngine.Tensor(CUDA.CuArray(layer.beta.data))  : nothing
            new_conv = ConvolutionalLayers.Conv2D(weights_gpu, bias_gpu, layer.stride, layer.padding, layer.use_batchnorm, gamma_gpu, beta_gpu)
            push!(layers_gpu, new_conv)
        elseif layer isa ConvKernelLayers.ConvKernelLayer
            new_conv = ConvKernelLayers.ConvKernelLayer(layer.in_channels, layer.out_channels, layer.kernel_size,
                                       layer.stride, layer.padding,
                                       TensorEngine.to_gpu(layer.weights),
                                       TensorEngine.to_gpu(layer.bias),
                                       TensorEngine.to_gpu(layer.gradW),
                                       TensorEngine.to_gpu(layer.gradB))
            push!(layers_gpu, new_conv)
        elseif layer isa ConvolutionalLayers.MaxPooling || layer isa GlobalAvgPool || layer isa Flatten
            push!(layers_gpu, layer)
        elseif layer isa Reshape
            push!(layers_gpu, Reshape(layer.output_shape))
        elseif layer isa Layers.DropoutLayer
            # DropoutLayer no tiene parámetros que mover
            push!(layers_gpu, layer)
        elseif layer isa Layers.BatchNorm
            # Obtener el número de canales de gamma
            channels = length(layer.gamma.data)
            
            # Crear nuevo BatchNorm con el mismo número de canales y configuración
            new_bn = Layers.BatchNorm(channels, 
                                    momentum=layer.momentum, 
                                    epsilon=layer.epsilon, 
                                    training=layer.training)
            
            # Reemplazar gamma y beta con versiones en GPU
            new_bn.gamma = TensorEngine.Tensor(CUDA.CuArray(layer.gamma.data))
            new_bn.beta = TensorEngine.Tensor(CUDA.CuArray(layer.beta.data))
            
            # Copiar running_mean y running_var (se quedan en CPU)
            new_bn.running_mean = copy(layer.running_mean)
            new_bn.running_var = copy(layer.running_var)
            
            push!(layers_gpu, new_bn)
        elseif layer isa Layers.ResidualBlock
            # Procesar las capas del camino convolucional recursivamente
            conv_path_gpu = AbstractLayer.Layer[]
            for sublayer in layer.conv_path
                temp_model = Sequential([sublayer])
                temp_gpu = model_to_gpu(temp_model)  # Llamada recursiva
                push!(conv_path_gpu, temp_gpu.layers[1])
            end
            
            # Procesar las capas del shortcut recursivamente
            shortcut_gpu = AbstractLayer.Layer[]
            for sublayer in layer.shortcut
                temp_model = Sequential([sublayer])
                temp_gpu = model_to_gpu(temp_model)  # Llamada recursiva
                push!(shortcut_gpu, temp_gpu.layers[1])
            end
            
            # Crear nuevo bloque residual con capas en GPU
            new_residual = Layers.ResidualBlock(conv_path_gpu, shortcut_gpu)
            push!(layers_gpu, new_residual)
        elseif layer isa Activation
            push!(layers_gpu, Activation(layer.f))
        elseif layer isa Layers.LayerActivation
            push!(layers_gpu, Layers.LayerActivation(layer.f))
        else
            error("Tipo de capa no soportado: $(typeof(layer))")
        end
    end
    return Sequential(layers_gpu)
end

end # module NeuralNetwork
