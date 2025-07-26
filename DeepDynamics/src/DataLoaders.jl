module DataLoaders

using Random
using CUDA
using Base.Threads
using ..TensorEngine
using ..GPUMemoryManager

export DataLoader, optimized_data_loader, cleanup_data_loader!

"""
    DataLoader(data, labels, batch_size; shuffle=true)

Crea un iterador que devuelve batches de datos y etiquetas.
Opcionalmente baraja los datos en cada época.
"""
mutable struct DataLoader
    data::Vector
    labels::Vector
    batch_size::Int
    shuffle::Bool
    indices::Vector{Int}
    current_epoch::Int
    
    function DataLoader(data, labels, batch_size; shuffle=true)
        @assert length(data) == length(labels) "Data y labels deben tener el mismo tamaño"
        indices = collect(1:length(data))
        if shuffle
            Random.shuffle!(indices)
        end
        return new(data, labels, batch_size, shuffle, indices, 0)
    end
end

function Base.iterate(dl::DataLoader, state=1)
    if state > length(dl.data)
        # Nueva época
        dl.current_epoch += 1
        if dl.shuffle
            Random.shuffle!(dl.indices)
        end
        return nothing
    end
    
    end_idx = min(state + dl.batch_size - 1, length(dl.data))
    batch_indices = dl.indices[state:end_idx]
    
    # Preparar batch
    batch_data = [dl.data[i] for i in batch_indices]
    batch_labels = [dl.labels[i] for i in batch_indices]
    
    return (batch_data, batch_labels), end_idx + 1
end

Base.length(dl::DataLoader) = Int(ceil(length(dl.data) / dl.batch_size))

# Estructura para manejar data loaders optimizados con limpieza
mutable struct OptimizedDataLoader
    channel::Channel{Tuple}
    task::Task
    is_active::Bool
end

"""
    optimized_data_loader(images, labels, batch_size; shuffle=true, to_gpu=true, prefetch=2)

Crea un generador de batches optimizado para GPU con prefetching y limpieza automática.
"""
function optimized_data_loader(images, labels, batch_size; shuffle=true, to_gpu=true, prefetch=2)
    n = length(images)
    indices = shuffle ? Random.shuffle!(collect(1:n)) : collect(1:n)
    
    # Prefetch buffers con mayor capacidad para evitar bloqueos
    prefetch_queue = Channel{Tuple}(prefetch)
    
    # Flag para controlar el estado del productor
    should_stop = Ref(false)
    
    # Crear función para procesar un batch con manejo de errores
    function process_batch(batch_indices)
        try
            # Extraer imágenes y etiquetas del batch
            batch_images = [images[j] for j in batch_indices]
            batch_labels = [labels[j] for j in batch_indices]
            
            # Mover a GPU si es necesario
            if to_gpu && CUDA.functional()
                batch_images = [TensorEngine.to_gpu(img) for img in batch_images]
                batch_labels = [TensorEngine.to_gpu(lbl) for lbl in batch_labels]
            end
            
            # Adaptar imágenes para el modelo
            batch_images = [
                ndims(img.data) == 3 ? adapt_image(img) : img
                for img in batch_images
            ]
            
            # Apilar
            stacked_images = stack_batch(batch_images)
            stacked_labels = stack_batch(batch_labels)
            
            return stacked_images, stacked_labels
        catch e
            @warn "Error procesando batch: $e"
            rethrow(e)
        end
    end
    
    # Productor en tarea asíncrona con manejo robusto de errores
    producer_task = @async begin
        try
            for i in 1:batch_size:n
                # Verificar si debemos detenernos
                if should_stop[]
                    break
                end
                
                end_idx = min(i + batch_size - 1, n)
                batch_indices = indices[i:end_idx]
                
                # Procesar batch
                batch = process_batch(batch_indices)
                
                # Enviar a la cola con timeout para evitar bloqueos
                if !put_with_timeout!(prefetch_queue, batch, 30.0)
                    @warn "Timeout enviando batch al canal"
                    break
                end
                
                # Liberar memoria GPU periódicamente
                if i % (batch_size * 10) == 0 && CUDA.functional()
                    GPUMemoryManager.check_and_clear_gpu_memory()
                end
            end
        catch e
            @error "Error en productor de datos: $e"
        finally
            close(prefetch_queue)
        end
    end
    
    # Retornar estructura que permite limpieza
    return OptimizedDataLoader(prefetch_queue, producer_task, true)
end

"""
    put_with_timeout!(channel::Channel, item, timeout)

Intenta poner un item en el canal con timeout.
"""
function put_with_timeout!(channel::Channel, item, timeout)
    deadline = time() + timeout
    while time() < deadline
        if isready(channel) || !isopen(channel)
            return false
        end
        try
            put!(channel, item)
            return true
        catch e
            if isa(e, InvalidStateException)
                return false
            end
            sleep(0.01)
        end
    end
    return false
end

"""
    cleanup_data_loader!(loader::OptimizedDataLoader)

Limpia recursos asociados con el data loader optimizado.
"""
function cleanup_data_loader!(loader::OptimizedDataLoader)
    if loader.is_active
        loader.is_active = false
        
        # Cerrar canal si está abierto
        if isopen(loader.channel)
            close(loader.channel)
        end
        
        # Esperar a que termine la tarea con timeout
        try
            wait_with_timeout(loader.task, 5.0)
        catch e
            @warn "Timeout esperando finalización del data loader"
        end
        
        # Limpiar memoria GPU
        if CUDA.functional()
            GPUMemoryManager.clear_cache()
        end
    end
end

"""
    wait_with_timeout(task::Task, timeout)

Espera a que termine una tarea con timeout.
"""
function wait_with_timeout(task::Task, timeout)
    deadline = time() + timeout
    while !istaskdone(task) && time() < deadline
        sleep(0.01)
    end
    if !istaskdone(task)
        throw(TimeoutError("Task did not complete within timeout"))
    end
end

# Iterador para OptimizedDataLoader
Base.iterate(loader::OptimizedDataLoader) = iterate(loader.channel)
Base.iterate(loader::OptimizedDataLoader, state) = iterate(loader.channel, state)

# Función auxiliar para adaptar imágenes
function adapt_image(img::TensorEngine.Tensor)
    data = img.data
    if ndims(data) == 3
        c, h, w = size(data)
        data_reshaped = reshape(data, (1, c, h, w))
    else
        data_reshaped = data
    end
    return TensorEngine.Tensor(data_reshaped; requires_grad=img.requires_grad)
end

# Stack batch con soporte mejorado
function stack_batch(tensors::Vector{<:TensorEngine.Tensor})
    if isempty(tensors)
        error("No se pueden apilar 0 tensores")
    end
    
    # Detectar si estamos en GPU
    is_on_gpu = tensors[1].data isa CUDA.CuArray
    
    # Determinar dimensiones
    nd = ndims(tensors[1].data)
    
    if nd == 4  # Imágenes NCHW
        if is_on_gpu
            # Versión GPU optimizada
            return TensorEngine.Tensor(cat([t.data for t in tensors]..., dims=1))
        else
            return TensorEngine.Tensor(cat([t.data for t in tensors]..., dims=1))
        end
    elseif nd == 1 || nd == 2  # Etiquetas
        if is_on_gpu
            return TensorEngine.Tensor(cat([t.data for t in tensors]..., dims=ndims(tensors[1].data)))
        else
            return TensorEngine.Tensor(hcat([vec(t.data) for t in tensors]...))
        end
    else
        error("Formato no soportado: ndims=$nd")
    end
end

# Función para limpiar todos los data loaders activos
const ACTIVE_LOADERS = OptimizedDataLoader[]

function register_loader!(loader::OptimizedDataLoader)
    push!(ACTIVE_LOADERS, loader)
end

function cleanup_all_loaders!()
    for loader in ACTIVE_LOADERS
        cleanup_data_loader!(loader)
    end
    empty!(ACTIVE_LOADERS)
end

# Asegurar limpieza al salir
atexit(cleanup_all_loaders!)

end # module