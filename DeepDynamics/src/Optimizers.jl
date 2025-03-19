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

# Optimizer Structures (keep existing implementation)
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

# Constructors (keep existing implementation)
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

# Optimizer Update Steps with GPU-Safe Operations
function step!(optimizer::SGD, parameters::Vector{TensorEngine.Tensor})
    for param in parameters
        if param.grad !== nothing
            param.data .= _safe_gpu_subtract(
                param.data, 
                _safe_gpu_multiply(optimizer.learning_rate, param.grad.data)
            )
        end
    end
end

function step!(optimizer::Adam, parameters::Vector{TensorEngine.Tensor})
    optimizer.t += 1
    
    # Calcular factores de corrección
    correction1 = Float32(1.0 - optimizer.beta1^optimizer.t)
    correction2 = Float32(1.0 - optimizer.beta2^optimizer.t)
    
    for param in parameters
        if param.grad === nothing
            continue
        end
        
        # Determinar si estamos en GPU o CPU
        is_gpu = param.data isa CuArray
        
        # Inicializar momentos si no existen
        if !haskey(optimizer.m, param)
            zeros_fn = is_gpu ? CUDA.zeros : zeros
            optimizer.m[param] = TensorEngine.Tensor(zeros_fn(Float32, size(param.data)))
            optimizer.v[param] = TensorEngine.Tensor(zeros_fn(Float32, size(param.data)))
        end
        
        # Asegurar que momentos estén en el mismo dispositivo que los parámetros
        if (optimizer.m[param].data isa CuArray) != is_gpu
            if is_gpu
                optimizer.m[param].data = CuArray(optimizer.m[param].data)
                optimizer.v[param].data = CuArray(optimizer.v[param].data)
            else
                optimizer.m[param].data = Array(optimizer.m[param].data)
                optimizer.v[param].data = Array(optimizer.v[param].data)
            end
        end
        
        # Referencias para mayor claridad
        mt = optimizer.m[param].data
        vt = optimizer.v[param].data
        grad = param.data
        
        # Actualización de parámetros sin mezclado de CPU/GPU
        if is_gpu
            # Versión GPU
            # Usar operaciones simples en lugar de broadcast complejo
            beta1 = Float32(optimizer.beta1)
            beta2 = Float32(optimizer.beta2)
            
            # Momento 1
            mt = beta1 .* mt .+ (1.0f0 - beta1) .* param.grad.data
            
            # Momento 2 (evitamos .^2 que puede ser problemático)
            grad_squared = param.grad.data .* param.grad.data
            vt = beta2 .* vt .+ (1.0f0 - beta2) .* grad_squared
            
            # Corregir bias
            mt_hat = mt ./ correction1
            vt_hat = vt ./ correction2
            
            # Actualizar parámetros
            param.data = param.data .- optimizer.learning_rate .* mt_hat ./ (sqrt.(vt_hat) .+ optimizer.epsilon)
            
            # Weight decay
            if optimizer.weight_decay > 0.0f0
                param.data = param.data .- optimizer.learning_rate .* optimizer.weight_decay .* param.data
            end
        else
            # Versión CPU
            beta1 = Float32(optimizer.beta1)
            beta2 = Float32(optimizer.beta2)
            
            # Momento 1
            mt = beta1 .* mt .+ (1.0f0 - beta1) .* param.grad.data
            
            # Momento 2
            grad_squared = param.grad.data .* param.grad.data
            vt = beta2 .* vt .+ (1.0f0 - beta2) .* grad_squared
            
            # Corregir bias
            mt_hat = mt ./ correction1
            vt_hat = vt ./ correction2
            
            # Actualizar parámetros
            param.data = param.data .- optimizer.learning_rate .* mt_hat ./ (sqrt.(vt_hat) .+ optimizer.epsilon)
            
            # Weight decay
            if optimizer.weight_decay > 0.0f0
                param.data = param.data .- optimizer.learning_rate .* optimizer.weight_decay .* param.data
            end
        end
        
        # Guardar momentos actualizados
        optimizer.m[param].data = mt
        optimizer.v[param].data = vt
    end
end

# Similar implementations for RMSProp, Adagrad, and Nadam
function step!(optimizer::RMSProp, parameters::Vector{TensorEngine.Tensor})
    for param in parameters
        if param.grad === nothing
            continue
        end
        
        # Initialize cache
        if !haskey(optimizer.cache, param)
            optimizer.cache[param] = TensorEngine.Tensor(zeros(Float32, size(param.data)))
        end
        
        cache_t = optimizer.cache[param]
        
        # Update cache
        cache_t.data .= _safe_gpu_multiply(optimizer.decay_rate, cache_t.data) .+ 
            _safe_gpu_multiply((1.0f0 - optimizer.decay_rate), 
                param.grad.data .^ 2)
        
        # Compute update
        update = param.grad.data ./ (sqrt.(cache_t.data) .+ optimizer.epsilon)
        
        # Update parameters
        param.data .= _safe_gpu_subtract(
            param.data, 
            _safe_gpu_multiply(optimizer.learning_rate, update)
        )
    end
end

function step!(optimizer::Adagrad, parameters::Vector{TensorEngine.Tensor})
    for param in parameters
        if param.grad === nothing
            continue
        end
        
        # Initialize cache
        if !haskey(optimizer.cache, param)
            optimizer.cache[param] = TensorEngine.Tensor(zeros(Float32, size(param.data)))
        end
        
        cache_t = optimizer.cache[param]
        
        # Update cache
        cache_t.data .+= param.grad.data .^ 2
        
        # Compute update
        update = param.grad.data ./ (sqrt.(cache_t.data) .+ optimizer.epsilon)
        
        # Update parameters
        param.data .= _safe_gpu_subtract(
            param.data, 
            _safe_gpu_multiply(optimizer.learning_rate, update)
        )
    end
end

function step!(optimizer::Nadam, parameters::Vector{TensorEngine.Tensor})
    optimizer.t += 1
    
    correction1 = 1.0f0 - optimizer.beta1^optimizer.t
    correction2 = 1.0f0 - optimizer.beta2^optimizer.t
    
    for param in parameters
        if param.grad === nothing
            continue
        end
        
        # Initialize momentum buffers
        if !haskey(optimizer.m, param)
            optimizer.m[param] = TensorEngine.Tensor(zeros(Float32, size(param.data)))
            optimizer.v[param] = TensorEngine.Tensor(zeros(Float32, size(param.data)))
        end
        
        mt = optimizer.m[param]
        vt = optimizer.v[param]
        
        # Update first moment estimate
        mt.data .= _safe_gpu_multiply(optimizer.beta1, mt.data) .+ 
                   _safe_gpu_multiply((1.0f0 - optimizer.beta1), param.grad.data)
        
        # Update second moment estimate
        vt.data .= _safe_gpu_multiply(optimizer.beta2, vt.data) .+ 
                   _safe_gpu_multiply((1.0f0 - optimizer.beta2), 
                       param.grad.data .^ 2)
        
        # Compute bias-corrected estimates
        mt_hat = mt.data ./ correction1
        vt_hat = vt.data ./ correction2
        
        # Nesterov-accelerated gradient
        nesterov = _safe_gpu_multiply(optimizer.beta1, mt_hat) .+ 
            _safe_gpu_multiply((1.0f0 - optimizer.beta1), 
                param.grad.data ./ correction1)
        
        # Compute update
        update = nesterov ./ (sqrt.(vt_hat) .+ optimizer.epsilon)
        
        # Apply weight decay
        if optimizer.weight_decay > 0.0f0
            update .+= _safe_gpu_multiply(optimizer.weight_decay, param.data)
        end
        
        # Update parameters
        param.data .= _safe_gpu_subtract(
            param.data, 
            _safe_gpu_multiply(optimizer.learning_rate, update)
        )
    end
end

end  # module Optimizers