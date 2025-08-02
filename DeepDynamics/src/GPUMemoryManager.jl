module GPUMemoryManager

using CUDA
export get_tensor_buffer, check_and_clear_gpu_memory, release_tensor_buffer, clear_cache, print_cache_stats, memory_stats

# Caché para tensores, organizada por tamaño y tipo
const TENSOR_CACHE = Dict{Tuple{Int, DataType}, Vector{CuArray}}()

# Estadísticas de uso
const CACHE_STATS = Dict{Symbol, Int}(
    :hits => 0,
    :misses => 0,
    :releases => 0
)

"""
    get_tensor_buffer(shape, type=Float32)

Obtiene un buffer GPU del tamaño y tipo especificados, reutilizando del caché si es posible.
"""
function get_tensor_buffer(shape, type=Float32)
    key = (prod(shape), type)
    try
        if haskey(TENSOR_CACHE, key) && !isempty(TENSOR_CACHE[key])
            buffer = pop!(TENSOR_CACHE[key])
            CACHE_STATS[:hits] += 1
            return reshape(buffer, shape)
        else
            CACHE_STATS[:misses] += 1
            return CUDA.zeros(type, shape)
        end
    catch e
        @warn "Failed to allocate GPU buffer: $e. Returning empty array."
        return CUDA.zeros(type, 0, 0, 0, 0)
    end
end

"""
    release_tensor_buffer(buffer)

Devuelve un buffer al caché para reutilización futura.
"""
function release_tensor_buffer(buffer::CuArray)
    try
        key = (length(buffer), eltype(buffer))
        if !haskey(TENSOR_CACHE, key)
            TENSOR_CACHE[key] = CuArray[]
        end
        push!(TENSOR_CACHE[key], buffer)
        CACHE_STATS[:releases] += 1
    catch e
        @warn "Failed to release GPU buffer: $e"
    end
end

"""
    clear_cache()

Libera toda la memoria almacenada en caché y fuerza limpieza del pool de CUDA.
"""
function clear_cache()
    try
        empty!(TENSOR_CACHE)
        GC.gc()
        CUDA.reclaim()
        for key in keys(CACHE_STATS)
            CACHE_STATS[key] = 0
        end
        #@info "GPU memory cache cleared"
    catch e
        @warn "Failed to clear GPU cache: $e"
    end
end

"""
    print_cache_stats()

Muestra estadísticas sobre el uso del caché de memoria.
"""
function print_cache_stats()
    total_bytes = 0
    counts = Dict{DataType, Int}()
    
    for ((size, type), buffers) in TENSOR_CACHE
        count = length(buffers)
        bytes = size * sizeof(type) * count
        counts[type] = get(counts, type, 0) + count
        total_bytes += bytes
    end
    
    println("=== GPU Cache Stats ===")
    println("Total cached: $(round(total_bytes / 1024^2, digits=2)) MB")
    for (type, count) in counts
        println("  $type: $count buffers")
    end
    println("Cache hits: $(CACHE_STATS[:hits])  misses: $(CACHE_STATS[:misses])  releases: $(CACHE_STATS[:releases])")
end

"""
    memory_stats()

Devuelve un resumen del estado actual de la memoria GPU.
"""
function memory_stats()
    if CUDA.functional()
        try
            # Usar CUDA.memory_status sin argumentos
            info = CUDA.memory_status()
            
            # Calcular valores
            total_bytes = info.total
            free_bytes = info.free
            used_bytes = total_bytes - free_bytes
            
            total_gb = total_bytes / 1e9
            used_gb = used_bytes / 1e9
            free_gb = free_bytes / 1e9
            free_percent = 100 * free_gb / total_gb
            
            return (
                total = total_gb,
                used = used_gb,
                free = free_gb,
                free_percent = free_percent
            )
        catch e
            # Si falla, retornar valores seguros
            @debug "Could not fetch GPU memory stats: $e"
            return (
                total = 1.0,
                used = 0.0,
                free = 1.0,
                free_percent = 100.0
            )
        end
    else
        return (
            total = 0.0,
            used = 0.0,
            free = 0.0,
            free_percent = 0.0
        )
    end
end


"""
    auto_clear_threshold = 0.85

Umbral de memoria GPU utilizada para activar limpieza automática.
"""
const auto_clear_threshold = 0.85

"""
    check_and_clear_gpu_memory(; verbose=false)

Verifica el uso de memoria GPU y limpia automáticamente si supera el umbral.
"""
function check_and_clear_gpu_memory(; verbose=false)
    if !CUDA.functional()
        return
    end
    
    try
        stats = memory_stats()
        
        # Verificar si tenemos estadísticas válidas
        if stats.total > 0
            used_fraction = stats.used / stats.total
            
            if used_fraction > auto_clear_threshold
                verbose && @info "GPU memory usage at $(round(used_fraction*100, digits=1))%. Clearing cache..."
                clear_cache()
                
                # Verificar después de limpiar
                new_stats = memory_stats()
                if new_stats.total > 0
                    new_used_fraction = new_stats.used / new_stats.total
                    verbose && @info "GPU memory after clearing: $(round(new_used_fraction*100, digits=1))%"
                end
            end
        end
    catch e
        @debug "Error checking GPU memory: $e"
    end
end



end # module
