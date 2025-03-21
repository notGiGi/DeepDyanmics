module TensorEngine

using ..GPUMemoryManager

export Tensor, add, matmul, backward, step!, mse_loss, initialize_grad!,
       initialize_weights, l2_regularization, compute_loss_with_regularization,
       clip_gradients!, to_gpu, to_cpu, softmax

using Random, Statistics, LinearAlgebra
try
    using CUDA
catch err
    @warn "CUDA not available; GPU support will not function."
end

# ---------------------------------------------------------------------------
# Definición mutable de la estructura Tensor de N dimensiones.
# Ahora es mutable para permitir modificar el campo grad.
# ---------------------------------------------------------------------------
mutable struct Tensor{N}
    data::AbstractArray{Float32, N}    # Datos en Float32
    grad::Union{Tensor{N}, Nothing}    # Gradiente como otro Tensor o nothing
    backward_fn::Union{Function, Nothing}
end

# ---------------------------------------------------------------------------
# Métodos de construcción
# ---------------------------------------------------------------------------
# Si se pasa ya un Tensor, se retorna el mismo.
Tensor(x::Tensor) = x

# Constructor para arrays que ya son Array{Float32, N}.
Tensor(data::AbstractArray{Float32, N}) where {N} = Tensor{N}(data, nothing, nothing)

# Constructor para arrays de otro tipo (se convierten a Float32).
function Tensor(data::AbstractArray{T, N}) where {T<:Real, N}
    return Tensor{N}(Float32.(data), nothing, nothing)
end

# ---------------------------------------------------------------------------
# Compatibilidad con Base
# ---------------------------------------------------------------------------
Base.size(t::Tensor) = size(t.data)
Base.ndims(t::Tensor) = ndims(t.data)

# ---------------------------------------------------------------------------
# Operaciones: add, matmul, backward, mse_loss, initialize_grad!, etc.
# ---------------------------------------------------------------------------
function ensure_compatible(data, reference_data)
    # Determinar tipo y dispositivo objetivo
    target_type = eltype(reference_data)
    target_on_gpu = reference_data isa CUDA.CuArray
    
    # Verificar si ya es compatible
    data_type = eltype(data)
    data_on_gpu = data isa CUDA.CuArray
    
    # Si ya es compatible, retornar sin cambios
    if data_type == target_type && data_on_gpu == target_on_gpu
        return data
    end
    
    # Convertir al tipo correcto primero
    if data_type != target_type
        if data_on_gpu
            data = CUDA.convert.(target_type, data)
        else
            data = convert.(target_type, data)
        end
    end
    
    # Luego transferir al dispositivo correcto
    if data_on_gpu != target_on_gpu
        if target_on_gpu
            data = CUDA.CuArray(data)
        else
            data = Array(data)
        end
    end
    
    return data
end


# Añade estas funciones a TensorEngine.jl
function device_of(x::AbstractArray)
    return x isa CUDA.CuArray ? :gpu : :cpu
end

function device_of(t::Tensor)
    return device_of(t.data)
end

function ensure_on_device(data, device::Symbol)
    current_device = device_of(data)
    if current_device == device
        return data
    end
    
    if device == :gpu
        return CUDA.CuArray(data)
    else
        return Array(data)
    end
end

function ensure_on_device(t::Tensor, device::Symbol)
    if device_of(t) == device
        return t
    end
    
    new_data = ensure_on_device(t.data, device)
    result = Tensor(new_data)
    
    if t.grad !== nothing
        result.grad = ensure_on_device(t.grad, device)
    end
    
    result.backward_fn = t.backward_fn
    return result
end



# Versión mejorada de add que maneja tipos flotantes
function add(t1::Tensor{N}, t2::Tensor{N}) where {N}
    # Determinar qué tensor tiene más prioridad (GPU sobre CPU, mayor precisión)
    reference = if t1.data isa CUDA.CuArray
        t1.data
    elseif t2.data isa CUDA.CuArray
        t2.data
    else
        # Si ambos están en CPU, usar el de mayor precisión
        if eltype(t1.data) == Float64 || eltype(t2.data) == Float64
            Float64.(t1.data)
        else
            t1.data
        end
    end
    
    # Asegurar compatibilidad
    t1_data = ensure_compatible(t1.data, reference)
    t2_data = ensure_compatible(t2.data, reference)
    
    # Realizar la operación
    result_data = t1_data .+ t2_data
    result = Tensor(result_data)
    
    # Definir backward
    result.backward_fn = grad -> begin
        backward(t1, Tensor(ensure_compatible(grad, t1.data)))
        backward(t2, Tensor(ensure_compatible(grad, t2.data)))
    end
    
    return result
end

# Versión mejorada de matmul que maneja tipos flotantes
function matmul(t1::Tensor{2}, t2::Tensor{2})::Tensor{2}
    # Determinar qué tensor tiene más prioridad
    reference = if t1.data isa CUDA.CuArray
        t1.data
    elseif t2.data isa CUDA.CuArray
        t2.data
    else
        # Si ambos están en CPU, usar el de mayor precisión
        if eltype(t1.data) == Float64 || eltype(t2.data) == Float64
            Float64.(t1.data)
        else
            t1.data
        end
    end
    
    # Asegurar compatibilidad
    t1_data = ensure_compatible(t1.data, reference)
    t2_data = ensure_compatible(t2.data, reference)
    
    # Realizar la multiplicación
    result_data = t1_data * t2_data
    result = Tensor(result_data)
    
    # Definir backward
    result.backward_fn = grad -> begin
        grad_compatible = ensure_compatible(grad, reference)
        t1_grad = grad_compatible * t2_data'
        t2_grad = t1_data' * grad_compatible
        
        backward(t1, Tensor(ensure_compatible(t1_grad, t1.data)))
        backward(t2, Tensor(ensure_compatible(t2_grad, t2.data)))
    end
    
    return result
end

# Reemplaza la función backward en TensorEngine.jl con esta versión

function backward(t::Tensor, grad::Tensor)
    # Detectar si el tensor t está en GPU
    t_on_gpu = (t.data isa CUDA.CuArray)
    
    # Detectar si el gradiente está en GPU
    grad_on_gpu = (grad.data isa CUDA.CuArray)
    
    # Obtener el tipo de datos del tensor t
    t_data_type = eltype(t.data)
    
    # Asegurarnos de que el gradiente esté en el mismo dispositivo y tipo que t
    grad_data = grad.data
    
    # Manejar la conversión de dispositivos y tipos
    if t_on_gpu && !grad_on_gpu
        # Si t está en GPU pero grad no, convertir grad a GPU y al tipo correcto
        grad_data = CUDA.CuArray{t_data_type}(convert.(t_data_type, grad_data))
    elseif !t_on_gpu && grad_on_gpu
        # Si t está en CPU pero grad no, convertir grad a CPU y al tipo correcto
        grad_data = Array{t_data_type}(convert.(t_data_type, Array(grad_data)))
    elseif eltype(grad_data) != t_data_type
        # Si ya están en el mismo dispositivo pero con tipos diferentes
        if t_on_gpu
            grad_data = CUDA.CuArray{t_data_type}(convert.(t_data_type, Array(grad_data)))
        else
            grad_data = convert.(t_data_type, grad_data)
        end
    end
    
    # Actualizar o inicializar el gradiente acumulado
    if t.grad === nothing
        # Si el gradiente es nil, inicializarlo
        t.grad = Tensor(grad_data)
    else
        # Asegurarnos de que t.grad está en el mismo dispositivo que grad_data
        t_grad_on_gpu = (t.grad.data isa CUDA.CuArray)
        t_grad_type = eltype(t.grad.data)
        
        # Manejar diferencias de dispositivo
        if t_grad_on_gpu != (grad_data isa CUDA.CuArray)
            if grad_data isa CUDA.CuArray
                t.grad.data = CUDA.CuArray{t_grad_type}(convert.(t_grad_type, Array(t.grad.data)))
            else
                t.grad.data = convert.(t_grad_type, Array(t.grad.data))
            end
        end
        
        # Manejar diferencias de tipo
        if eltype(t.grad.data) != eltype(grad_data)
            if t_grad_on_gpu
                t.grad.data = CUDA.CuArray{t_grad_type}(convert.(t_grad_type, Array(grad_data)))
            else
                grad_data = convert.(t_grad_type, grad_data)
            end
        end
        
        # Acumular el gradiente
        t.grad.data .+= grad_data
    end
    
    # Llamar a la función de retropropagación si existe
    if t.backward_fn !== nothing
        t.backward_fn(grad_data)
    end
end

function mse_loss(y_pred::Tensor, y_true::Tensor)::Tensor
    error = y_pred.data .- y_true.data
    loss_val = sum(error .^ 2) / max(length(y_pred.data), 1)
    result = Tensor(reshape([loss_val], (1,1)))
    result.backward_fn = _ -> begin
        grad_input = 2 .* error ./ max(length(y_pred.data), 1)
        backward(y_pred, Tensor(grad_input))
    end
    return result
end

function initialize_grad!(t::Tensor)
    # Crea un Tensor de ceros del mismo tamaño que t.data y lo asigna a t.grad.
    t.grad = Tensor(zeros(Float32, size(t.data)))
end

# ---------------------------------------------------------------------------
# Inicialización de pesos y regularización
# ---------------------------------------------------------------------------
function initialize_weights(size::Tuple{Int,Int}; method::Symbol = :xavier)::Tensor
    fan_in, fan_out = size
    scale = Float32(
        if method == :he
            sqrt(2.0f0 / fan_in)
        elseif method == :xavier
            sqrt(1.0f0 / (fan_in + fan_out))
        else
            0.01f0
        end
    )
    return Tensor(scale .* randn(Float32, fan_out, fan_in))
end

function l2_regularization(weights::Vector{Tensor}, λ::Float64)::Tensor
    reg = λ * sum(sum(w.data .^ 2) for w in weights)
    return Tensor([reg])
end

function compute_loss_with_regularization(output::Tensor, target::Tensor, weights::Vector{Tensor}, λ::Float64)::Tensor
    mse = mse_loss(output, target)
    reg_term = isempty(weights) ? Tensor([0.0]) : l2_regularization(weights, λ)
    total_loss = Tensor(mse.data .+ reg_term.data)
    return total_loss
end

function clip_gradients!(t::Tensor, threshold::Float64)
    if t.grad !== nothing
        grad_norm = norm(t.grad.data)
        if grad_norm > threshold
            scaling_factor = threshold / grad_norm
            t.grad.data .*= scaling_factor
        end
    end
end

# ---------------------------------------------------------------------------
# Paso de actualización (optimización)
# ---------------------------------------------------------------------------
function step!(optimizer, parameters::Vector{Tensor})
    if optimizer isa SGD
        for param in parameters
            if param.grad !== nothing
                @inbounds param.data .-= optimizer.learning_rate .* param.grad.data
            end
        end
    elseif optimizer isa Adam
        optimizer.t += 1
        for param in parameters
            if param.grad === nothing
                continue
            end
            if !haskey(optimizer.m, param)
                optimizer.m[param] = Tensor(zeros(Float32, size(param.data)))
                optimizer.v[param] = Tensor(zeros(Float32, size(param.data)))
            end
            mt = optimizer.m[param]
            vt = optimizer.v[param]
            @inbounds mt.data .= optimizer.beta1 .* mt.data .+ (1.0 - optimizer.beta1) .* param.grad.data
            @inbounds vt.data .= optimizer.beta2 .* vt.data .+ (1.0 - optimizer.beta2) .* (param.grad.data .^ 2)
            mt_hat = mt.data ./ (1.0 - optimizer.beta1^optimizer.t)
            vt_hat = vt.data ./ (1.0 - optimizer.beta2^optimizer.t)
            @inbounds param.data .-= optimizer.learning_rate .* (mt_hat ./ (sqrt.(vt_hat) .+ optimizer.epsilon))
        end
    else
        error("Optimizer not implemented")
    end
end

# ---------------------------------------------------------------------------
# Función Softmax
# ---------------------------------------------------------------------------
function softmax(x::Tensor)::Tensor
    exps = exp.(x.data .- maximum(x.data))
    sum_exps = sum(exps)
    probs = exps ./ sum_exps
    return Tensor(probs)
end

# ---------------------------------------------------------------------------
# Funciones para mover tensores a GPU/CPU
# ---------------------------------------------------------------------------

function to_gpu(t::TensorEngine.Tensor)
    if CUDA.functional()
        if !(t.data isa CuArray)
            # Asegurarnos de que los datos están en GPU
            t.data = CuArray(t.data)
        end
        if t.grad !== nothing && !(t.grad.data isa CuArray)
            # También convertir el gradiente a GPU
            t.grad.data = CuArray(t.grad.data)
        end
    else
        @warn "CUDA not functional, tensor remains on CPU."
    end
    return t
end

function to_gpu(x::CUDA.CuArray)
    # Si el objeto ya es un CuArray, simplemente lo retorna.
    return x
end

function to_cpu(t::Tensor)
    t.data = Array(t.data)
    if t.grad !== nothing
        t.grad.data = Array(t.grad.data)
    end
    return t
end

# ---------------------------------------------------------------------------
# Manejo de Memoria para GPU
# ---------------------------------------------------------------------------
const MEMORY_POOL = Dict{Tuple, CUDA.CuArray}()

function get_buffer(shape, dtype=Float32)
    key = (shape, dtype)
    if !haskey(MEMORY_POOL, key)
        MEMORY_POOL[key] = CUDA.zeros(dtype, shape)
    end
    return MEMORY_POOL[key]
end

function release_buffers!()
    empty!(MEMORY_POOL)
    CUDA.reclaim()
end


# Versión optimizada para crear tensores en GPU
function tensor_gpu(shape, type=Float32)
    buffer = GPUMemoryManager.get_tensor_buffer(shape, type)
    return Tensor(buffer)
end

# Liberar tensor GPU cuando ya no es necesario
function release_tensor(tensor::Tensor)
    if tensor.data isa CuArray
        GPUMemoryManager.release_tensor_buffer(tensor.data)
    end
end


end  # module TensorEngine


