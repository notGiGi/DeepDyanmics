module Optimizers

using CUDA
using LinearAlgebra

# Assuming TensorEngine is imported or available in the parent module
using ..TensorEngine

export SGD, Adam, RMSProp, Adagrad, Nadam, step!

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

# ========== FASE 2: Optimizer Update Steps con mejoras ==========

# SGD con verificación de NaN
function step!(optimizer::SGD, parameters::Vector{<:TensorEngine.Tensor})
    for param in parameters
        if param.grad !== nothing
            # NUEVO: Verificar NaN
            if any(isnan.(param.grad.data))
                @warn "NaN detectado en gradientes, saltando actualización"
                continue
            end
            
            # Actualización simple SGD
            @inbounds param.data .-= optimizer.learning_rate .* param.grad.data
            
            # Verificar NaN después de actualizar
            if any(isnan.(param.data))
                error("NaN en parámetros después de SGD update")
            end
        end
    end
end

# Adam con bias correction y manejo de dispositivos
function step!(optimizer::Adam, parameters::Vector{<:TensorEngine.Tensor})
    optimizer.t += 1
    
    # Calcular factores de corrección de bias
    bias_correction1 = 1.0f0 - optimizer.beta1^optimizer.t
    bias_correction2 = 1.0f0 - optimizer.beta2^optimizer.t
    
    for param in parameters
        # Skip si no hay gradiente
        if param.grad === nothing
            continue
        end
        
        # NUEVO: Verificar NaN antes de procesar
        if any(isnan.(param.grad.data))
            @warn "NaN detectado en gradientes, saltando actualización del parámetro"
            continue
        end
        
        # Detectar dispositivo del parámetro
        is_gpu = param.data isa CuArray
        
        # Inicializar momentos si no existen
        if !haskey(optimizer.m, param)
            # CAMBIO 2.3: Crear momentos en el mismo dispositivo que el parámetro
            if is_gpu
                optimizer.m[param] = TensorEngine.Tensor(CUDA.zeros(Float32, size(param.data)); requires_grad=false)
                optimizer.v[param] = TensorEngine.Tensor(CUDA.zeros(Float32, size(param.data)); requires_grad=false)
            else
                optimizer.m[param] = TensorEngine.Tensor(zeros(Float32, size(param.data)); requires_grad=false)
                optimizer.v[param] = TensorEngine.Tensor(zeros(Float32, size(param.data)); requires_grad=false)
            end
        end
        
        # Asegurar que momentos estén en el mismo dispositivo que los parámetros
        mt = optimizer.m[param]
        vt = optimizer.v[param]
        
        # Verificar consistencia de dispositivos
        if (mt.data isa CuArray) != is_gpu
            if is_gpu
                mt.data = CuArray(mt.data)
                vt.data = CuArray(vt.data)
            else
                mt.data = Array(mt.data)
                vt.data = Array(vt.data)
            end
        end
        
        # Referencias para mayor claridad
        grad = param.grad.data
        
        # Actualizar primer momento (media)
        @inbounds mt.data .= optimizer.beta1 .* mt.data .+ (1.0f0 - optimizer.beta1) .* grad
        
        # Actualizar segundo momento (varianza no centrada)
        @inbounds vt.data .= optimizer.beta2 .* vt.data .+ (1.0f0 - optimizer.beta2) .* (grad .^ 2)
        
        # Calcular estimados corregidos por bias
        mt_hat = mt.data ./ bias_correction1
        vt_hat = vt.data ./ bias_correction2
        
        # Actualizar parámetros
        @inbounds param.data .-= optimizer.learning_rate .* (mt_hat ./ (sqrt.(vt_hat) .+ optimizer.epsilon))
        
        # Weight decay si está configurado
        if optimizer.weight_decay > 0.0f0
            @inbounds param.data .-= optimizer.learning_rate .* optimizer.weight_decay .* param.data
        end
        
        # NUEVO: Verificar NaN después de la actualización
        if any(isnan.(param.data))
            error("NaN detectado en parámetros después de actualización. Entrenamiento divergió.")
        end
    end
end

# RMSProp mejorado
function step!(optimizer::RMSProp, parameters::Vector{<:TensorEngine.Tensor})
    for param in parameters
        if param.grad === nothing
            continue
        end
        
        # Verificar NaN
        if any(isnan.(param.grad.data))
            @warn "NaN en gradientes, saltando"
            continue
        end
        
        # Detectar dispositivo
        is_gpu = param.data isa CuArray
        
        # Inicializar cache si no existe
        if !haskey(optimizer.cache, param)
            if is_gpu
                optimizer.cache[param] = TensorEngine.Tensor(CUDA.zeros(Float32, size(param.data)); requires_grad=false)
            else
                optimizer.cache[param] = TensorEngine.Tensor(zeros(Float32, size(param.data)); requires_grad=false)
            end
        end
        
        cache = optimizer.cache[param]
        
        # Verificar dispositivo del cache
        if (cache.data isa CuArray) != is_gpu
            if is_gpu
                cache.data = CuArray(cache.data)
            else
                cache.data = Array(cache.data)
            end
        end
        
        # Actualizar cache (media móvil del cuadrado de gradientes)
        @inbounds cache.data .= optimizer.decay_rate .* cache.data .+ 
                               (1.0f0 - optimizer.decay_rate) .* (param.grad.data .^ 2)
        
        # Actualizar parámetros
        @inbounds param.data .-= optimizer.learning_rate .* param.grad.data ./ 
                                (sqrt.(cache.data) .+ optimizer.epsilon)
    end
end

# Adagrad (manteniendo la funcionalidad original pero con verificación de NaN)
function step!(optimizer::Adagrad, parameters::Vector{<:TensorEngine.Tensor})
    for param in parameters
        if param.grad === nothing
            continue
        end
        
        # Verificar NaN
        if any(isnan.(param.grad.data))
            @warn "NaN en gradientes, saltando"
            continue
        end
        
        # Detectar dispositivo
        is_gpu = param.data isa CuArray
        
        # Initialize cache
        if !haskey(optimizer.cache, param)
            if is_gpu
                optimizer.cache[param] = TensorEngine.Tensor(CUDA.zeros(Float32, size(param.data)); requires_grad=false)
            else
                optimizer.cache[param] = TensorEngine.Tensor(zeros(Float32, size(param.data)); requires_grad=false)
            end
        end
        
        cache_t = optimizer.cache[param]
        
        # Verificar dispositivo del cache
        if (cache_t.data isa CuArray) != is_gpu
            if is_gpu
                cache_t.data = CuArray(cache_t.data)
            else
                cache_t.data = Array(cache_t.data)
            end
        end
        
        # Update cache
        cache_t.data .+= param.grad.data .^ 2
        
        # Compute update
        update = param.grad.data ./ (sqrt.(cache_t.data) .+ optimizer.epsilon)
        
        # Update parameters
        param.data .-= optimizer.learning_rate .* update
    end
end

# Nadam con mejoras
function step!(optimizer::Nadam, parameters::Vector{<:TensorEngine.Tensor})
    optimizer.t += 1
    
    correction1 = 1.0f0 - optimizer.beta1^optimizer.t
    correction2 = 1.0f0 - optimizer.beta2^optimizer.t
    
    for param in parameters
        if param.grad === nothing
            continue
        end
        
        # Verificar NaN
        if any(isnan.(param.grad.data))
            @warn "NaN en gradientes, saltando"
            continue
        end
        
        # Detectar dispositivo
        is_gpu = param.data isa CuArray
        
        # Initialize momentum buffers
        if !haskey(optimizer.m, param)
            if is_gpu
                optimizer.m[param] = TensorEngine.Tensor(CUDA.zeros(Float32, size(param.data)); requires_grad=false)
                optimizer.v[param] = TensorEngine.Tensor(CUDA.zeros(Float32, size(param.data)); requires_grad=false)
            else
                optimizer.m[param] = TensorEngine.Tensor(zeros(Float32, size(param.data)); requires_grad=false)
                optimizer.v[param] = TensorEngine.Tensor(zeros(Float32, size(param.data)); requires_grad=false)
            end
        end
        
        mt = optimizer.m[param]
        vt = optimizer.v[param]
        
        # Verificar dispositivos
        if (mt.data isa CuArray) != is_gpu
            if is_gpu
                mt.data = CuArray(mt.data)
                vt.data = CuArray(vt.data)
            else
                mt.data = Array(mt.data)
                vt.data = Array(vt.data)
            end
        end
        
        # Update first moment estimate
        mt.data .= optimizer.beta1 .* mt.data .+ (1.0f0 - optimizer.beta1) .* param.grad.data
        
        # Update second moment estimate
        vt.data .= optimizer.beta2 .* vt.data .+ (1.0f0 - optimizer.beta2) .* (param.grad.data .^ 2)
        
        # Compute bias-corrected estimates
        mt_hat = mt.data ./ correction1
        vt_hat = vt.data ./ correction2
        
        # Nesterov-accelerated gradient
        nesterov = optimizer.beta1 .* mt_hat .+ (1.0f0 - optimizer.beta1) .* param.grad.data ./ correction1
        
        # Compute update
        update = nesterov ./ (sqrt.(vt_hat) .+ optimizer.epsilon)
        
        # Apply weight decay
        if optimizer.weight_decay > 0.0f0
            update .+= optimizer.weight_decay .* param.data
        end
        
        # Update parameters
        param.data .-= optimizer.learning_rate .* update
        
        # Verificar NaN después
        if any(isnan.(param.data))
            error("NaN detectado en parámetros después de actualización Nadam")
        end
    end
end

end  # module Optimizers