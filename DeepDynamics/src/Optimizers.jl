module Optimizers

using CUDA
using LinearAlgebra

# Assuming TensorEngine is imported or available in the parent module
using ..TensorEngine

export SGD, Adam, RMSProp, Adagrad, Nadam, step!, clip_gradients_norm!

"""
    create_zeros_like(tensor::Tensor)

Crea un tensor de ceros en el mismo dispositivo y tipo que `tensor`
"""
function create_zeros_like(tensor::TensorEngine.Tensor)
    shape = size(tensor.data)
    device = TensorEngine.device_of(tensor)
    data = device == :gpu ? CUDA.zeros(Float32, shape) : zeros(Float32, shape)
    return TensorEngine.Tensor(data; requires_grad=false)
end





# Utility function to ensure GPU-compatible operations
function _ensure_gpu_array(x)
    if x isa CuArray
        return x
    elseif x isa AbstractArray
        return CuArray(x)
    else
        error("Cannot convert $(typeof(x)) to CuArray")
    end
end

# Wrapper for GPU-safe operations
function _gpu_safe_operation(op::Function, a, b)
    a_gpu = _ensure_gpu_array(a)
    b_gpu = _ensure_gpu_array(b)
    return op(a_gpu, b_gpu)
end

# Custom GPU-safe multiplication
function _gpu_multiply(a, b)
    _gpu_safe_operation(.*,  a, b)
end

# Custom GPU-safe subtraction
function _gpu_subtract(a, b)
    _gpu_safe_operation(.-,  a, b)
end

# Utility functions for safe GPU operations
function _safe_gpu_multiply(a, b)
    # Handles different input types
    if a isa Number && b isa CuArray
        return a .* b
    elseif a isa CuArray && b isa Number
        return a .* b
    elseif a isa CuArray && b isa CuArray
        return a .* b
    elseif a isa CuArray
        return a .* CuArray(b)
    elseif b isa CuArray
        return CuArray(a) .* b
    else
        return a .* b
    end
end

function _safe_gpu_subtract(a, b)
    # Handles different input types
    if a isa CuArray && b isa CuArray
        return a .- b
    elseif a isa CuArray
        return a .- CuArray(b)
    elseif b isa CuArray
        return CuArray(a) .- b
    else
        return a .- b
    end
end

# Optimizer Structures
mutable struct SGD
    learning_rate::Float32
end

mutable struct Adam
    learning_rate::Float32
    beta1::Float32
    beta2::Float32
    epsilon::Float32
    weight_decay::Float32
    t::Int
    m::Dict{TensorEngine.Tensor, TensorEngine.Tensor}
    v::Dict{TensorEngine.Tensor, TensorEngine.Tensor}
end

mutable struct RMSProp
    learning_rate::Float32
    decay_rate::Float32
    epsilon::Float32
    cache::Dict{TensorEngine.Tensor, TensorEngine.Tensor}
end

mutable struct Adagrad
    learning_rate::Float32
    epsilon::Float32
    cache::Dict{TensorEngine.Tensor, TensorEngine.Tensor}
end

mutable struct Nadam
    learning_rate::Float32
    beta1::Float32
    beta2::Float32
    epsilon::Float32
    weight_decay::Float32
    t::Int
    m::Dict{TensorEngine.Tensor, TensorEngine.Tensor}
    v::Dict{TensorEngine.Tensor, TensorEngine.Tensor}
end

# Constructors
function SGD(; learning_rate::Real = 0.01)
    return SGD(Float32(learning_rate))
end

function Adam(;
    learning_rate::Real = 0.001, 
    beta1::Real = 0.9, 
    beta2::Real = 0.999, 
    epsilon::Real = 1e-8,
    weight_decay::Real = 0.0
)
    return Adam(
        Float32(learning_rate), 
        Float32(beta1), 
        Float32(beta2), 
        Float32(epsilon), 
        Float32(weight_decay),
        0, 
        Dict{TensorEngine.Tensor, TensorEngine.Tensor}(), 
        Dict{TensorEngine.Tensor, TensorEngine.Tensor}()
    )
end

# Constructor alternativo para compatibilidad con argumentos posicionales
function Adam(learning_rate::Real)
    return Adam(; learning_rate=learning_rate)
end


function RMSProp(; 
    learning_rate::Real = 0.001, 
    decay_rate::Real = 0.9, 
    epsilon::Real = 1e-8
)
    return RMSProp(
        Float32(learning_rate), 
        Float32(decay_rate), 
        Float32(epsilon), 
        Dict{TensorEngine.Tensor, TensorEngine.Tensor}()
    )
end

function Adagrad(; 
    learning_rate::Real = 0.01, 
    epsilon::Real = 1e-8
)
    return Adagrad(
        Float32(learning_rate), 
        Float32(epsilon), 
        Dict{TensorEngine.Tensor, TensorEngine.Tensor}()
    )
end

function Nadam(;
    learning_rate::Real = 0.002, 
    beta1::Real = 0.9, 
    beta2::Real = 0.999, 
    epsilon::Real = 1e-8,
    weight_decay::Real = 0.0
)
    return Nadam(
        Float32(learning_rate), 
        Float32(beta1), 
        Float32(beta2), 
        Float32(epsilon), 
        Float32(weight_decay),
        0, 
        Dict{TensorEngine.Tensor, TensorEngine.Tensor}(), 
        Dict{TensorEngine.Tensor, TensorEngine.Tensor}()
    )
end

"""
    clip_gradients_norm!(params, max_norm)
    
Gradient clipping por norma global.
CRÍTICO para RNN, LSTM, GRU para prevenir exploding gradients.
"""
function clip_gradients_norm!(params::Vector{<:TensorEngine.Tensor}, max_norm::Float32=5.0f0)
    total_norm = 0f0
    
    # Calcular norma L2 global
    for p in params
        if p.grad !== nothing
            grad_data = p.grad.data
            total_norm += sum(abs2, grad_data)
        end
    end
    
    total_norm = sqrt(total_norm)
    
    # Aplicar clipping si excede max_norm
    if total_norm > max_norm
        clip_coef = max_norm / (total_norm + 1f-6)
        for p in params
            if p.grad !== nothing
                p.grad.data .*= clip_coef
            end
        end
    end
    
    return total_norm
end


# ========== FASE 2: Optimizer Update Steps con mejoras ==========

# ========== Optimizer Update Steps con sincronización automática de momentos ==========

# SGD
function step!(optimizer::SGD, parameters::Vector{<:TensorEngine.Tensor})
    for param in parameters
        if param.grad !== nothing
            if any(isnan.(param.grad.data))
                @warn "NaN detected in gradients, skipping update"
                continue
            end
            param.data .-= optimizer.learning_rate .* param.grad.data
            if any(isnan.(param.data))
                error("NaN detected in parameters after SGD update")
            end
        end
    end
end

# Adam
function step!(optimizer::Adam, parameters::Vector{<:TensorEngine.Tensor}; 
               clip_norm::Union{Nothing,Float32}=nothing)
    # Aplicar gradient clipping si se especifica
    if clip_norm !== nothing
        norm = clip_gradients_norm!(parameters, clip_norm)
        if norm > clip_norm * 2
            @warn "Gradient explosion detected: norm=$norm, clipped to $clip_norm"
        end
    end
    
    # CÓDIGO EXISTENTE DE ADAM (sin cambios)
    optimizer.t += 1
    bias_correction1 = 1.0f0 - optimizer.beta1^optimizer.t
    bias_correction2 = 1.0f0 - optimizer.beta2^optimizer.t

    for param in parameters
        if param.grad === nothing
            continue
        end

        if any(isnan.(param.grad.data))
            @warn "NaN detected in gradients, skipping update"
            continue
        end

        if !haskey(optimizer.m, param)
            optimizer.m[param] = create_zeros_like(param)
            optimizer.v[param] = create_zeros_like(param)
        end

        # Sincronizar momentos al dispositivo del parámetro
        optimizer.m[param] = TensorEngine.ensure_on_device(optimizer.m[param], TensorEngine.device_of(param))
        optimizer.v[param] = TensorEngine.ensure_on_device(optimizer.v[param], TensorEngine.device_of(param))

        mt = optimizer.m[param]
        vt = optimizer.v[param]
        grad = param.grad.data

        mt.data .= optimizer.beta1 .* mt.data .+ (1.0f0 - optimizer.beta1) .* grad
        vt.data .= optimizer.beta2 .* vt.data .+ (1.0f0 - optimizer.beta2) .* (grad .^ 2)

        m_hat = mt.data ./ bias_correction1
        v_hat = vt.data ./ bias_correction2

        update = optimizer.learning_rate .* m_hat ./ (sqrt.(v_hat) .+ optimizer.epsilon)

        if optimizer.weight_decay > 0.0f0
            update .+= optimizer.weight_decay .* param.data
        end

        param.data .-= update

        if any(isnan.(param.data))
            error("NaN detected in parameters after Adam update")
        end
    end
end

# RMSProp
function step!(optimizer::RMSProp, parameters::Vector{<:TensorEngine.Tensor})
    for param in parameters
        if param.grad === nothing
            continue
        end

        if any(isnan.(param.grad.data))
            @warn "NaN detected in gradients, skipping update"
            continue
        end

        if !haskey(optimizer.cache, param)
            optimizer.cache[param] = create_zeros_like(param)
        end

        # Sincronizar cache al dispositivo del parámetro
        optimizer.cache[param] = TensorEngine.ensure_on_device(optimizer.cache[param], TensorEngine.device_of(param))

        cache = optimizer.cache[param]
        cache.data .= optimizer.decay_rate .* cache.data .+ (1.0f0 - optimizer.decay_rate) .* (param.grad.data .^ 2)

        update = optimizer.learning_rate .* param.grad.data ./ (sqrt.(cache.data) .+ optimizer.epsilon)
        param.data .-= update
    end
end

# Adagrad
function step!(optimizer::Adagrad, parameters::Vector{<:TensorEngine.Tensor})
    for param in parameters
        if param.grad === nothing
            continue
        end

        if any(isnan.(param.grad.data))
            @warn "NaN detected in gradients, skipping update"
            continue
        end

        if !haskey(optimizer.cache, param)
            optimizer.cache[param] = create_zeros_like(param)
        end

        # Sincronizar cache al dispositivo del parámetro
        optimizer.cache[param] = TensorEngine.ensure_on_device(optimizer.cache[param], TensorEngine.device_of(param))

        cache = optimizer.cache[param]
        cache.data .+= param.grad.data .^ 2

        update = optimizer.learning_rate .* param.grad.data ./ (sqrt.(cache.data) .+ optimizer.epsilon)
        param.data .-= update
    end
end

# Nadam
function step!(optimizer::Nadam, parameters::Vector{<:TensorEngine.Tensor})
    optimizer.t += 1
    correction1 = 1.0f0 - optimizer.beta1^optimizer.t
    correction2 = 1.0f0 - optimizer.beta2^optimizer.t

    for param in parameters
        if param.grad === nothing
            continue
        end

        if any(isnan.(param.grad.data))
            @warn "NaN detected in gradients, skipping update"
            continue
        end

        if !haskey(optimizer.m, param)
            optimizer.m[param] = create_zeros_like(param)
            optimizer.v[param] = create_zeros_like(param)
        end

        # Sincronizar momentos al dispositivo del parámetro
        optimizer.m[param] = TensorEngine.ensure_on_device(optimizer.m[param], TensorEngine.device_of(param))
        optimizer.v[param] = TensorEngine.ensure_on_device(optimizer.v[param], TensorEngine.device_of(param))

        mt = optimizer.m[param]
        vt = optimizer.v[param]
        grad = param.grad.data

        mt.data .= optimizer.beta1 .* mt.data .+ (1.0f0 - optimizer.beta1) .* grad
        vt.data .= optimizer.beta2 .* vt.data .+ (1.0f0 - optimizer.beta2) .* (grad .^ 2)

        m_hat = mt.data ./ correction1
        v_hat = vt.data ./ correction2

        nesterov = optimizer.beta1 .* m_hat .+ (1.0f0 - optimizer.beta1) .* grad ./ correction1

        update = optimizer.learning_rate .* nesterov ./ (sqrt.(v_hat) .+ optimizer.epsilon)

        if optimizer.weight_decay > 0.0f0
            update .+= optimizer.weight_decay .* param.data
        end

        param.data .-= update

        if any(isnan.(param.data))
            error("NaN detected in parameters after Nadam update")
        end
    end
end

"""
    optimizer_to_device!(opt, device::Symbol)

Mueve todo el estado interno (momentos, cache) del optimizador al dispositivo especificado.
"""
function optimizer_to_device!(opt, device::Symbol)
    for field in (:m, :v, :cache)
        if hasfield(typeof(opt), field)
            dict = getfield(opt, field)
            if dict !== nothing
                for (p, buf) in dict
                    dict[p] = TensorEngine.ensure_on_device(buf, device)
                end
            end
        end
    end

    return opt
end


end  # module Optimizers