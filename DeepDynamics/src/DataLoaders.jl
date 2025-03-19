module DataLoaders

using Random
using CUDA
using Base.Threads
using ..TensorEngine
using ..GPUMemoryManager

export DataLoader, optimized_data_loader, parallel_data_loader

"""
    DataLoader(data, labels, batch_size; shuffle=true)

Crea un iterador que devuelve batches de datos y etiquetas.
Opcionalmente baraja los datos en cada época.
"""
struct DataLoader
    data::Vector
    labels::Vector
    batch_size::Int
    shuffle::Bool
    indices::Vector{Int}
    
    function DataLoader(data, labels, batch_size; shuffle=true)
        @assert length(data) == length(labels) "Data y labels deben tener el mismo tamaño"
        indices = collect(1:length(data))
        if shuffle
            Random.shuffle!(indices)
        end
        return new(data, labels, batch_size, shuffle, indices)
    end
end

function Base.iterate(dl::DataLoader, state=1)
    if state > length(dl.data)
        # Si es necesario, re-barajar para la próxima época
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

"""
    optimized_data_loader(images, labels, batch_size; shuffle=true, to_gpu=true, prefetch=2)

Crea un generador de batches optimizado para GPU con prefetching.
"""
function optimized_data_loader(images, labels, batch_size; shuffle=true, to_gpu=true, prefetch=2)
    n = length(images)
    indices = shuffle ? Random.shuffle(1:n) : 1:n
    
    # Prefetch buffers (mantener varios batches en memoria)
    prefetch_queue = Channel{Tuple}(prefetch)
    
    # Crear función para procesar un batch
    function process_batch(batch_indices)
        # Extraer imágenes y etiquetas del batch
        batch_images = [images[j] for j in batch_indices]
        batch_labels = [labels[j] for j in batch_indices]
        
        # Mover a GPU si es necesario
        if to_gpu
            batch_images = [TensorEngine.to_gpu(img) for img in batch_images]
            batch_labels = [TensorEngine.to_gpu(lbl) for lbl in batch_labels]
        end
        
        # Adaptar imágenes para el modelo (si no están ya adaptadas)
        batch_images = [
            if ndims(img.data) == 3
                TensorEngine.adapt_image(img)
            else
                img
            end
            for img in batch_images
        ]
        
        # Apilar
        stacked_images = TensorEngine.stack_batch(batch_images)
        stacked_labels = TensorEngine.stack_batch(batch_labels)
        
        return stacked_images, stacked_labels
    end
    
    # Productor en hilo separado
    @async begin
        try
            for i in 1:batch_size:n
                end_idx = min(i + batch_size - 1, n)
                batch_indices = indices[i:end_idx]
                
                # Procesar batch
                batch = process_batch(batch_indices)
                
                # Enviar a la cola
                put!(prefetch_queue, batch)
            end
        finally
            close(prefetch_queue)
        end
    end
    
    return prefetch_queue
end

end # module