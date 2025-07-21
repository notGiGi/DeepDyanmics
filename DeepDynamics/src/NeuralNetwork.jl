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
    out = _new_activation_tensor(max.(t.data, 0f0), t)
    if out.requires_grad
        out.backward_fn = grad -> TensorEngine.backward(
            t,
            TensorEngine.Tensor((t.data .> 0f0) .* grad)
        )
    end
    return out
end

# ---------- Sigmoid ----------
function sigmoid(t::TensorEngine.Tensor)
    σ = 1f0 ./ (1f0 .+ exp.(-t.data))
    out = _new_activation_tensor(σ, t)
    if out.requires_grad
        out.backward_fn = grad -> TensorEngine.backward(
            t,
            TensorEngine.Tensor(σ .* (1f0 .- σ) .* grad)
        )
    end
    return out
end

# ---------- Tanh ----------
function tanh_activation(t::TensorEngine.Tensor)
    τ = tanh.(t.data)
    out = _new_activation_tensor(τ, t)
    if out.requires_grad
        out.backward_fn = grad -> TensorEngine.backward(
            t,
            TensorEngine.Tensor((1f0 .- τ.^2) .* grad)
        )
    end
    return out
end

# ---------- Leaky‑ReLU ----------
function leaky_relu(t::TensorEngine.Tensor; α = 0.01f0)
    y  = max.(t.data, α .* t.data)
    out = _new_activation_tensor(y, t)
    if out.requires_grad
        deriv = map(x -> x > 0f0 ? 1f0 : α, t.data)
        out.backward_fn = grad -> TensorEngine.backward(
            t,
            TensorEngine.Tensor(deriv .* grad)
        )
    end
    return out
end

# ---------- Swish ----------
function swish(t::TensorEngine.Tensor)
    σ = 1f0 ./ (1f0 .+ exp.(-t.data))
    y = t.data .* σ
    out = _new_activation_tensor(y, t)
    if out.requires_grad
        grad_factor = σ .+ t.data .* σ .* (1f0 .- σ)
        out.backward_fn = grad -> TensorEngine.backward(
            t,
            TensorEngine.Tensor(grad .* grad_factor)
        )
    end
    return out
end

# ---------- Mish ----------
function mish(t::TensorEngine.Tensor)
    sp = log.(1f0 .+ exp.(t.data))        # softplus
    τ  = tanh.(sp)
    y  = t.data .* τ
    out = _new_activation_tensor(y, t)d
    if out.requires_grad
        δ = τ .+ t.data .* (1f0 .- τ.^2) .* (exp.(t.data) ./ (1f0 .+ exp.(t.data)))
        out.backward_fn = grad -> TensorEngine.backward(
            t,
            TensorEngine.Tensor(grad .* δ)
        )
    end
    return out
end

# ---------- Softmax ----------
function softmax(t::TensorEngine.Tensor)::TensorEngine.Tensor
    max_val      = maximum(t.data)
    log_sum_exp  = log.(sum(exp.(t.data .- max_val))) + max_val
    probs        = exp.(t.data .- log_sum_exp)

    # ➊ conservar el flag
    out = TensorEngine.Tensor(probs; requires_grad = t.requires_grad)

    # ➋ mantener el backward
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

"""
    model_to_gpu(model::Sequential) → Sequential

Crea una copia del `Sequential` donde **todos los tensores que
requieren gradiente** pasan a `CuArray`, preservando su bandera
`requires_grad`.  
Los tensores que no requieren gradiente se copian igualmente para que
estén en la GPU (por coherencia de dispositivo), pero con
`requires_grad = false`.
"""
function model_to_gpu(model::Sequential)
    layers_gpu = Vector{AbstractLayer.Layer}()

    for layer in model.layers
        ##################################################################
        # 1. Dense
        ##################################################################
        if layer isa Dense
            Wg = TensorEngine.Tensor(
                     CUDA.CuArray(layer.weights.data);
                     requires_grad = layer.weights.requires_grad)

            bg = TensorEngine.Tensor(
                     CUDA.CuArray(layer.biases.data);
                     requires_grad = layer.biases.requires_grad)

            push!(layers_gpu, Dense(Wg, bg))

        ##################################################################
        # 2. Conv2D  (con o sin BatchNorm interno)
        ##################################################################
        elseif layer isa ConvolutionalLayers.Conv2D
            Wg = TensorEngine.Tensor(
                     CUDA.CuArray(layer.weights.data);
                     requires_grad = layer.weights.requires_grad)

            bg = TensorEngine.Tensor(
                     CUDA.CuArray(layer.bias.data);
                     requires_grad = layer.bias.requires_grad)

            γg = layer.use_batchnorm && layer.gamma !== nothing ?
                     TensorEngine.Tensor(
                         CUDA.CuArray(layer.gamma.data);
                         requires_grad = layer.gamma.requires_grad) : nothing

            βg = layer.use_batchnorm && layer.beta  !== nothing ?
                     TensorEngine.Tensor(
                         CUDA.CuArray(layer.beta.data);
                         requires_grad = layer.beta.requires_grad)  : nothing

            new_conv = ConvolutionalLayers.Conv2D(
                           Wg, bg, layer.stride, layer.padding,
                           layer.use_batchnorm, γg, βg)

            push!(layers_gpu, new_conv)

        ##################################################################
        # 3. ConvKernelLayer
        ##################################################################
        elseif layer isa ConvKernelLayers.ConvKernelLayer
            new_conv = ConvKernelLayers.ConvKernelLayer(
                           layer.in_channels, layer.out_channels, layer.kernel_size,
                           layer.stride, layer.padding,
                           TensorEngine.Tensor(
                               CUDA.CuArray(layer.weights.data);
                               requires_grad = layer.weights.requires_grad),
                           TensorEngine.Tensor(
                               CUDA.CuArray(layer.bias.data);
                               requires_grad = layer.bias.requires_grad),
                           TensorEngine.Tensor(
                               CUDA.CuArray(layer.gradW.data); requires_grad = false),
                           TensorEngine.Tensor(
                               CUDA.CuArray(layer.gradB.data); requires_grad = false))
            push!(layers_gpu, new_conv)

        ##################################################################
        # 4. BatchNorm
        ##################################################################
        elseif layer isa Layers.BatchNorm
            ch = length(layer.gamma.data)
            new_bn = Layers.BatchNorm(ch;
                                      momentum = layer.momentum,
                                      epsilon  = layer.epsilon,
                                      training = layer.training)

            #  ── copiar parámetros conservando requires_grad ────────────
            new_bn.gamma = TensorEngine.Tensor(
                               CUDA.CuArray(layer.gamma.data);
                               requires_grad = layer.gamma.requires_grad)

            new_bn.beta  = TensorEngine.Tensor(
                               CUDA.CuArray(layer.beta.data);
                               requires_grad = layer.beta.requires_grad)

            # running stats (buffers) siguen en CPU
            new_bn.running_mean .= layer.running_mean
            new_bn.running_var  .= layer.running_var
            new_bn.num_batches_tracked = layer.num_batches_tracked

            push!(layers_gpu, new_bn)

        ##################################################################
        # 5. Bloques residuales  (procesar recursivamente)
        ##################################################################
        elseif layer isa Layers.ResidualBlock
            conv_path_gpu = AbstractLayer.Layer[]
            for sub in layer.conv_path
                push!(conv_path_gpu,
                      model_to_gpu(Sequential([sub])).layers[1])
            end

            shortcut_gpu = AbstractLayer.Layer[]
            for sub in layer.shortcut
                push!(shortcut_gpu,
                      model_to_gpu(Sequential([sub])).layers[1])
            end

            push!(layers_gpu, Layers.ResidualBlock(conv_path_gpu, shortcut_gpu))

        ##################################################################
        # 6. Capas sin parámetros (se copian tal cual)
        ##################################################################
        elseif layer isa ConvolutionalLayers.MaxPooling ||
               layer isa GlobalAvgPool                    ||
               layer isa Flatten                          ||
               layer isa Reshape                          ||
               layer isa Layers.DropoutLayer              ||
               layer isa Activation                       ||
               layer isa Layers.LayerActivation
            push!(layers_gpu, layer)

        ##################################################################
        # 7. Tipo desconocido
        ##################################################################
        else
            error("Tipo de capa no soportado en model_to_gpu: $(typeof(layer))")
        end
    end

    return Sequential(layers_gpu)
end



end # module NeuralNetwork
