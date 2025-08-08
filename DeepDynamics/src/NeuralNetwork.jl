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
export Sequential, Dense, Activation, collect_parameters,
       relu, softmax, sigmoid, tanh_activation, leaky_relu,
       model_to_gpu, model_to_cpu, model_device, model_to_device,
       layer_to_device, forward
import ..Layers: BatchNorm
import ..Layers: forward as batchnorm_forward

# ==================================================================
# Capa Dense Optimizada (Sin activación integrada)
# ==================================================================
mutable struct Dense <: AbstractLayer.Layer
    weights::TensorEngine.Tensor
    biases::TensorEngine.Tensor
end

function Dense(in_dim::Integer, out_dim::Integer;
               init_method::Symbol = :xavier,
               requires_grad::Bool = true)
    # Elige estrategia de init
    W_data = if init_method == :he
        σ = sqrt(2f0 / in_dim)
        randn(Float32, out_dim, in_dim) .* σ
    elseif init_method == :xavier
        σ = sqrt(1f0 / in_dim)
        randn(Float32, out_dim, in_dim) .* σ
    else
        error("Unknown init_method: $init_method")
    end
    b_data = zeros(Float32, out_dim, 1)

    # Envuelve en Tensor
    W = Tensor(W_data; requires_grad=requires_grad)
    b = Tensor(b_data; requires_grad=requires_grad)

    return Dense(W, b)  # llama al inner constructor
end


function forward(layer::Dense, input::TensorEngine.Tensor)
    # Asegurar que datos están en el mismo dispositivo
    W = TensorEngine.ensure_on_device(layer.weights.data, TensorEngine.device_of(input))
    b = TensorEngine.ensure_on_device(layer.biases.data, TensorEngine.device_of(input))
    x = input.data

    # Verificar dimensiones: (input_size, batch_size)
    if size(x, 1) != size(W, 2)
        error("Dense layer expects input size $(size(W, 2)), got $(size(x, 1))")
    end

    # Calcular salida: W * x + b
    output = W * x .+ reshape(b, :, 1)

    # Crear tensor de salida
    out_tensor = TensorEngine.Tensor(output, requires_grad=input.requires_grad || layer.weights.requires_grad)

    # Definir backward
    out_tensor.backward_fn = function(grad)
        ∇W = grad * x'
        ∇b = sum(grad, dims=2)
        ∇x = W' * grad

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

function forward(model::Sequential, input::Tensor; verbose::Bool=false)
    current = input
    for (i, layer) in enumerate(model.layers)
        # Ejecutar la capa
        out = layer(current)

        # Validar salida
        if out isa Tensor
            current = out
        elseif out isa AbstractArray
            current = Tensor(out; requires_grad=current.requires_grad)
        else
            error("Layer $i returned unsupported type $(typeof(out))")
        end

        verbose && @info "Layer $i output size: $(size(current.data)) device: $(device_of(current))"
    end
    return current
end

# Sobrecarga de llamada para que funcione model(input)
(model::Sequential)(input::Tensor) = forward(model, input)



(model::Sequential)(input) = forward(model, input)




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
# Funciones de Activación — versión que preserva requires_grad
# ==================================================================

"""
    _new_activation_tensor(data, parent::Tensor)

Crea un Tensor que:
  • copia `data`  
  • hereda `requires_grad` de `parent`
"""
function _new_activation_tensor(data, parent::TensorEngine.Tensor)
    TensorEngine.Tensor(
        data;
        requires_grad = parent.requires_grad   #  ← CLAVE
    )
end

# ---------- ReLU ----------
function relu(t::TensorEngine.Tensor)
    # NO USAR t.data!
    output = max.(t, 0f0)  # Esto debería mantener el grafo
    # O si necesitas trabajar con arrays:
    out_data = max.(t.data, 0f0)
    out = TensorEngine.Tensor(out_data; requires_grad=t.requires_grad)
    if out.requires_grad
        out.backward_fn = function(grad)
            grad_input = (t.data .> 0f0) .* grad
            TensorEngine.backward(t, TensorEngine.Tensor(grad_input))
        end
    end
    return out
end

# ---------- Sigmoid ----------  
function sigmoid(t::TensorEngine.Tensor)
    neg_x = -t.data
    exp_neg_x = exp.(neg_x)
    σ_data = 1f0 ./ (1f0 .+ exp_neg_x)
    
    out = TensorEngine.Tensor(σ_data; requires_grad=t.requires_grad)
    
    if out.requires_grad
        # Capturar σ como tensor para que tenga .data
        σ = out
        out.backward_fn = function(grad)
            grad_data = σ.data .* (1f0 .- σ.data) .* grad
            TensorEngine.backward(t, TensorEngine.Tensor(grad_data))
        end
    end
    
    return out
end


# ---------- Tanh ----------
function tanh_activation(t::TensorEngine.Tensor)
    τ = tanh.(t.data)
    out = TensorEngine.Tensor(τ; requires_grad=t.requires_grad)
    if out.requires_grad
        out.backward_fn = function(grad)
            grad_input = (1f0 .- τ.^2) .* grad
            TensorEngine.backward(t, TensorEngine.Tensor(grad_input))
        end
    end
    return out
end

# ---------- Leaky ReLU ----------
function leaky_relu(t::TensorEngine.Tensor; α = 0.01f0)
    y = max.(t.data, α .* t.data)
    out = TensorEngine.Tensor(y; requires_grad=t.requires_grad)
    if out.requires_grad
        out.backward_fn = function(grad)
            deriv = map(x -> x > 0f0 ? 1f0 : α, t.data)
            grad_input = deriv .* grad
            TensorEngine.backward(t, TensorEngine.Tensor(grad_input))
        end
    end
    return out
end

# ---------- Swish ----------
function swish(t::TensorEngine.Tensor)
    neg_x = -t.data
    exp_neg_x = exp.(neg_x)
    σ = 1f0 ./ (1f0 .+ exp_neg_x)
    y = t.data .* σ
    
    out = TensorEngine.Tensor(y; requires_grad=t.requires_grad)
    if out.requires_grad
        out.backward_fn = function(grad)
            grad_factor = σ .+ t.data .* σ .* (1f0 .- σ)
            grad_input = grad .* grad_factor
            TensorEngine.backward(t, TensorEngine.Tensor(grad_input))
        end
    end
    return out
end

# ---------- Mish ----------
function mish(t::TensorEngine.Tensor)
    exp_x = exp.(t.data)
    sp = log.(1f0 .+ exp_x)  # softplus
    τ = tanh.(sp)
    y = t.data .* τ
    
    out = TensorEngine.Tensor(y; requires_grad=t.requires_grad)
    if out.requires_grad
        out.backward_fn = function(grad)
            σ = exp_x ./ (1f0 .+ exp_x)
            δ = τ .+ t.data .* (1f0 .- τ.^2) .* σ
            grad_input = grad .* δ
            TensorEngine.backward(t, TensorEngine.Tensor(grad_input))
        end
    end
    return out
end

# ---------- Softmax ---------- (esta ya está bien)
function softmax(t::TensorEngine.Tensor)::TensorEngine.Tensor
    max_vals = maximum(t.data, dims=1)
    exp_vals = exp.(t.data .- max_vals)
    sum_exp = sum(exp_vals, dims=1)
    probs = exp_vals ./ sum_exp
    
    out = TensorEngine.Tensor(probs; requires_grad=t.requires_grad)
    
    if out.requires_grad
        out.backward_fn = function(grad)
            grad_sum = sum(grad .* probs, dims=1)
            grad_input = probs .* (grad .- grad_sum)
            TensorEngine.backward(t, TensorEngine.Tensor(grad_input))
        end
    end
    
    return out
end


# ==================================================================
# Funciones de Ayuda
# ==================================================================
"""
    model_to_device(model::Sequential, device::Symbol) -> Sequential

Devuelve una copia del modelo con todos sus parámetros movidos al dispositivo especificado (:gpu o :cpu).
"""
function model_to_device(model::Sequential, device::Symbol)
    layers_new = [layer_to_device(layer, device) for layer in model.layers]
    return Sequential(layers_new)
end

"""
    model_to_gpu(model::Sequential) -> Sequential

Devuelve una copia del modelo en GPU. Si CUDA no está disponible, devuelve el modelo original.
"""
function model_to_gpu(model::Sequential)
    if !CUDA.functional()
        @warn "CUDA no está disponible. El modelo se mantiene en CPU."
        return model
    end
    return model_to_device(model, :gpu)
end

"""
    model_to_cpu(model::Sequential) -> Sequential

Devuelve una copia del modelo en CPU.
"""
function model_to_cpu(model::Sequential)
    return model_to_device(model, :cpu)
end

"""
    model_device(model::Sequential) -> Symbol

Detecta en qué dispositivo está el modelo (basado en su primer parámetro encontrado).
"""
function model_device(model::Sequential)
    params = collect_parameters(model)
    if isempty(params)
        return :cpu  # Si no hay parámetros asumimos CPU
    end
    return TensorEngine.device_of(params[1])
end

"""
    collect_parameters(model::Sequential) -> Vector{Tensor}

Devuelve todos los tensores entrenables del modelo (pesos y bias).
"""
function collect_parameters(model::Sequential)
    params = TensorEngine.Tensor[]
    for layer in model.layers
        append!(params, collect_layer_parameters(layer))
    end
    return params
end

"""
    collect_layer_parameters(layer) -> Vector{Tensor}

Devuelve todos los tensores entrenables de una capa individual.
"""
function collect_layer_parameters(layer)
    params = TensorEngine.Tensor[]
    
    if layer isa Dense
        for p in (layer.weights, layer.biases)
            push!(params, p isa Tensor ? p : Tensor(p; requires_grad=true))
        end
    elseif layer isa Layers.BatchNorm
        for p in (layer.gamma, layer.beta)
            push!(params, p isa Tensor ? p : Tensor(p; requires_grad=true))
        end
    elseif layer isa ConvolutionalLayers.Conv2D
        for p in (layer.weights, layer.bias)
            push!(params, p isa Tensor ? p : Tensor(p; requires_grad=true))
        end
        if layer.use_batchnorm && layer.gamma !== nothing
            for p in (layer.gamma, layer.beta)
                push!(params, p isa Tensor ? p : Tensor(p; requires_grad=true))
            end
        end
    elseif layer isa ConvKernelLayers.ConvKernelLayer
        for p in (layer.weights, layer.bias)
            push!(params, p isa Tensor ? p : Tensor(p; requires_grad=true))
        end
    elseif layer isa EmbeddingLayer.Embedding
        p = layer.weights
        push!(params, p isa Tensor ? p : Tensor(p; requires_grad=true))
    elseif layer isa Layers.ResidualBlock
        for sub in layer.conv_path
            append!(params, collect_layer_parameters(sub))
        end
        for sub in layer.shortcut
            append!(params, collect_layer_parameters(sub))
        end
    end
    
    return params
end


"""
    layer_to_device(layer, device::Symbol)

Mueve una capa individual al dispositivo especificado.
"""
function layer_to_device(layer::Dense, device::Symbol)
    weights_new = TensorEngine.ensure_on_device(layer.weights, device)
    biases_new  = TensorEngine.ensure_on_device(layer.biases, device)
    return Dense(weights_new, biases_new)
end

function layer_to_device(layer::Layers.BatchNorm, device::Symbol)
    ch = length(layer.gamma.data)
    bn_new = Layers.BatchNorm(
        ch; momentum=layer.momentum,
            epsilon=layer.epsilon,
            training=layer.training
    )
    bn_new.gamma = TensorEngine.ensure_on_device(layer.gamma, device)
    bn_new.beta  = TensorEngine.ensure_on_device(layer.beta, device)
    bn_new.running_mean .= layer.running_mean
    bn_new.running_var  .= layer.running_var
    bn_new.num_batches_tracked = layer.num_batches_tracked
    return bn_new
end

function layer_to_device(layer::ConvolutionalLayers.Conv2D, device::Symbol)
    weights_new = TensorEngine.ensure_on_device(layer.weights, device)
    bias_new    = TensorEngine.ensure_on_device(layer.bias, device)
    gamma_new   = layer.gamma !== nothing ? 
                  TensorEngine.ensure_on_device(layer.gamma, device) : nothing
    beta_new    = layer.beta  !== nothing ? 
                  TensorEngine.ensure_on_device(layer.beta, device)  : nothing
    return ConvolutionalLayers.Conv2D(
        weights_new, bias_new, layer.stride, layer.padding,
        layer.use_batchnorm, gamma_new, beta_new
    )
end

function layer_to_device(layer::ConvKernelLayers.ConvKernelLayer, device::Symbol)
    weights_new = TensorEngine.ensure_on_device(layer.weights, device)
    bias_new    = TensorEngine.ensure_on_device(layer.bias, device)
    gradW_new   = TensorEngine.ensure_on_device(layer.gradW, device)
    gradB_new   = TensorEngine.ensure_on_device(layer.gradB, device)
    return ConvKernelLayers.ConvKernelLayer(
        layer.in_channels, layer.out_channels, layer.kernel_size,
        layer.stride, layer.padding,
        weights_new, bias_new, gradW_new, gradB_new
    )
end

function layer_to_device(layer::ResidualBlock, device::Symbol)
    conv_path_new = Vector{AbstractLayer.Layer}()
    for sublayer in layer.conv_path
        push!(conv_path_new, layer_to_device(sublayer, device))
    end

    shortcut_new = Vector{AbstractLayer.Layer}()
    for sublayer in layer.shortcut
        push!(shortcut_new, layer_to_device(sublayer, device))
    end

    return ResidualBlock(conv_path_new, shortcut_new)
end




function layer_to_device(layer, device::Symbol)
    # Capas sin parámetros (Flatten, Activation, Dropout, etc.)
    return layer
end

"""
    forward(layer::DropoutLayer, input::TensorEngine.Tensor)

Aplica dropout durante entrenamiento, pasa sin cambios durante evaluación.
"""
function NeuralNetwork.forward(layer::DropoutLayer, input::TensorEngine.Tensor)
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

"""
    zero_grad!(model::Sequential)

Reinicia los gradientes de todos los parámetros del modelo a cero.
"""
function zero_grad!(model::Sequential)
    params = collect_parameters(model)
    for p in params
        if p.requires_grad
            TensorEngine.zero_grad!(p)
        end
    end
end


function forward(layer::BatchNorm, input::Tensor)
    return batchnorm_forward(layer, input)
end


end # module NeuralNetwork
