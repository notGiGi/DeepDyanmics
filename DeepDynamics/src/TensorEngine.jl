module TensorEngine

using ..GPUMemoryManager

export Tensor, add, matmul, backward, mse_loss, initialize_grad!,
       initialize_weights, l2_regularization, compute_loss_with_regularization,
       clip_gradients!, to_gpu, to_cpu, softmax, zero_grad!

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
    data::AbstractArray{Float32, N}
    grad::Union{Nothing, Tensor}  # CAMBIO: Quitar {N} para permitir cualquier dimensión
    backward_fn::Union{Function, Nothing}
    requires_grad::Bool
end

# ---------------------------------------------------------------------------
# Métodos de construcción
# ---------------------------------------------------------------------------
# Si se pasa ya un Tensor, se retorna el mismo.
Tensor(x::Tensor) = x

# Constructor para arrays que ya son Array{Float32, N}.
function Tensor(data::AbstractArray{Float32, N}; requires_grad=true) where {N}
    return Tensor{N}(data, nothing, nothing, requires_grad)
end

# Constructor para arrays de otro tipo (se convierten a Float32).
function Tensor(data::AbstractArray{T, N}; requires_grad=true) where {T<:Real, N}
    return Tensor{N}(Float32.(data), nothing, nothing, requires_grad)
end

# ---------------------------------------------------------------------------
# Compatibilidad con Base
# ---------------------------------------------------------------------------
Base.size(t::Tensor) = size(t.data)
Base.ndims(t::Tensor) = ndims(t.data)

# ---------------------------------------------------------------------------
# Función zero_grad!
# ---------------------------------------------------------------------------
"""
    zero_grad!(t::Tensor)

Reinicia los gradientes del tensor a cero. Esencial para evitar acumulación entre batches.
"""
function zero_grad!(t::Tensor)
    if t.grad !== nothing && t.grad.data !== nothing
        fill!(t.grad.data, 0f0)
    end
end

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
    result = Tensor(new_data; requires_grad=t.requires_grad)
    
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
    result = Tensor(result_data; requires_grad=(t1.requires_grad || t2.requires_grad))
    
    # Definir backward
    result.backward_fn = grad -> begin
        if t1.requires_grad
            backward(t1, Tensor(ensure_compatible(grad, t1.data); requires_grad=false))
        end
        if t2.requires_grad
            backward(t2, Tensor(ensure_compatible(grad, t2.data); requires_grad=false))
        end
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
    result = Tensor(result_data; requires_grad=(t1.requires_grad || t2.requires_grad))
    
    # Definir backward
    result.backward_fn = grad -> begin
        grad_compatible = ensure_compatible(grad, reference)
        if t1.requires_grad
            t1_grad = grad_compatible * t2_data'
            backward(t1, Tensor(ensure_compatible(t1_grad, t1.data); requires_grad=false))
        end
        if t2.requires_grad
            t2_grad = t1_data' * grad_compatible
            backward(t2, Tensor(ensure_compatible(t2_grad, t2.data); requires_grad=false))
        end
    end
    
    return result
end

# Agregar estas funciones a TensorEngine.jl después de la función add

# Multiplicación escalar
function Base.:*(scalar::Number, t::Tensor)
    result_data = scalar .* t.data
    result = Tensor(result_data; requires_grad=t.requires_grad)
    
    if t.requires_grad
        result.backward_fn = grad -> begin
            grad_data = grad isa Tensor ? grad.data : grad
            backward(t, Tensor(scalar .* grad_data; requires_grad=false))
        end
    end
    
    return result
end

function Base.:*(t::Tensor, scalar::Number)
    return scalar * t
end

# Suma escalar
function Base.:+(t::Tensor, scalar::Number)
    result_data = t.data .+ scalar
    result = Tensor(result_data; requires_grad=t.requires_grad)
    
    if t.requires_grad
        result.backward_fn = grad -> begin
            grad_data = grad isa Tensor ? grad.data : grad
            backward(t, Tensor(grad_data; requires_grad=false))
        end
    end
    
    return result
end

function Base.:+(scalar::Number, t::Tensor)
    return t + scalar
end

# Resta
function Base.:-(t1::Tensor{N}, t2::Tensor{N}) where {N}
    result_data = t1.data .- t2.data
    result = Tensor(result_data; requires_grad=(t1.requires_grad || t2.requires_grad))
    
    result.backward_fn = grad -> begin
        grad_data = grad isa Tensor ? grad.data : grad
        if t1.requires_grad
            backward(t1, Tensor(grad_data; requires_grad=false))
        end
        if t2.requires_grad
            backward(t2, Tensor(-grad_data; requires_grad=false))
        end
    end
    
    return result
end

# División
function Base.:/(t::Tensor, scalar::Number)
    return (1.0f0 / scalar) * t
end

# Para soportar broadcasting con .* .+ etc
Base.broadcastable(t::Tensor) = t
Base.axes(t::Tensor) = axes(t.data)
Base.ndims(::Type{<:Tensor{N}}) where {N} = N
Base.eltype(t::Tensor) = eltype(t.data)
Base.similar(t::Tensor, ::Type{T}, dims::Dims) where {T} = Tensor(similar(t.data, T, dims))

# Implementar interfaz de broadcasting
struct TensorStyle <: Base.Broadcast.BroadcastStyle end
Base.BroadcastStyle(::Type{<:Tensor}) = TensorStyle()

# Cuando se mezclan Tensor y Array/Number
Base.BroadcastStyle(::TensorStyle, ::Base.Broadcast.DefaultArrayStyle) = TensorStyle()
Base.BroadcastStyle(::TensorStyle, ::Base.Broadcast.Style{Tuple}) = TensorStyle()

# Materializar el broadcast
function Base.Broadcast.broadcasted(::TensorStyle, op, args...)
    # Extraer datos de los Tensores
    data_args = map(args) do arg
        arg isa Tensor ? arg.data : arg
    end
    
    # Aplicar la operación
    result_data = Base.Broadcast.broadcasted(op, data_args...)
    
    # Si es una operación simple, materializarla
    if op in (+, -, *, /)
        result_data = Base.Broadcast.materialize(result_data)
        # Determinar si requiere gradientes
        requires_grad = any(arg isa Tensor && arg.requires_grad for arg in args)
        return Tensor(result_data; requires_grad=requires_grad)
    else
        # Para operaciones más complejas, devolver el objeto broadcasted
        return result_data
    end
end

# Para que funcione el broadcast assignment
Base.copyto!(dest::Tensor, src::Tensor) = (copyto!(dest.data, src.data); dest)
Base.copyto!(dest::Tensor, src) = (copyto!(dest.data, src); dest)

# Reemplaza la función backward en TensorEngine.jl con esta versión
function backward(t::Tensor, grad::Union{Tensor, AbstractArray})
    # Verificar si requiere gradientes
    if !t.requires_grad
        return
    end
    
    # Convertir grad a datos si es Tensor
    grad_data = grad isa Tensor ? grad.data : grad
    
    # Detectar dispositivos
    t_on_gpu = (t.data isa CUDA.CuArray)
    grad_on_gpu = (grad_data isa CUDA.CuArray)
    
    # Obtener tipo de datos
    t_data_type = eltype(t.data)
    
    # Manejar conversión de dispositivos y tipos
    if t_on_gpu && !grad_on_gpu
        grad_data = CUDA.CuArray{t_data_type}(convert.(t_data_type, grad_data))
    elseif !t_on_gpu && grad_on_gpu
        grad_data = Array{t_data_type}(convert.(t_data_type, Array(grad_data)))
    elseif eltype(grad_data) != t_data_type
        if t_on_gpu
            grad_data = CUDA.CuArray{t_data_type}(convert.(t_data_type, Array(grad_data)))
        else
            grad_data = convert.(t_data_type, grad_data)
        end
    end
    
    # Verificar dimensiones y expandir si es necesario
    if size(grad_data) != size(t.data)
        # Si grad es escalar y t no, expandir grad
        if length(grad_data) == 1
            grad_data = fill(grad_data[1], size(t.data))
        else
            error("Gradient shape $(size(grad_data)) doesn't match tensor shape $(size(t.data))")
        end
    end
    
    # Actualizar o inicializar el gradiente acumulado
    if t.grad === nothing
        t.grad = Tensor(copy(grad_data); requires_grad=false)
    else
        # Acumular gradientes
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
    
    # Crear tensor escalar (1D con 1 elemento)
    result = Tensor([loss_val]; requires_grad=true)
    
    result.backward_fn = grad_scalar -> begin
        # grad_scalar es el gradiente que viene de arriba (escalar)
        grad_val = grad_scalar isa AbstractArray ? grad_scalar[1] : grad_scalar
        grad_input = (2.0f0 .* error ./ max(length(y_pred.data), 1)) .* grad_val
        backward(y_pred, Tensor(grad_input))
    end
    
    return result
end

function initialize_grad!(t::Tensor)
    if t.requires_grad && t.grad === nothing
        t.grad = Tensor(zeros(Float32, size(t.data)); requires_grad=false)
    end
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
    return Tensor(scale .* randn(Float32, fan_out, fan_in); requires_grad=true)
end

function l2_regularization(weights::Vector{Tensor}, λ::Float64)::Tensor
    reg = λ * sum(sum(w.data .^ 2) for w in weights)
    return Tensor([reg]; requires_grad=false)
end

function compute_loss_with_regularization(output::Tensor, target::Tensor, weights::Vector{Tensor}, λ::Float64)::Tensor
    mse = mse_loss(output, target)
    reg_term = isempty(weights) ? Tensor([0.0]; requires_grad=false) : l2_regularization(weights, λ)
    total_loss = Tensor(mse.data .+ reg_term.data; requires_grad=true)
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
# Función Softmax
# ---------------------------------------------------------------------------
function softmax(x::Tensor)::Tensor
    exps = exp.(x.data .- maximum(x.data))
    sum_exps = sum(exps)
    probs = exps ./ sum_exps
    return Tensor(probs; requires_grad=x.requires_grad)
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
    return Tensor(buffer; requires_grad=true)
end

# Liberar tensor GPU cuando ya no es necesario
function release_tensor(tensor::Tensor)
    if tensor.data isa CuArray
        GPUMemoryManager.release_tensor_buffer(tensor.data)
    end
end


end  # module TensorEngine