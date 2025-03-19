module GPUMemoryManager

using CUDA
export get_tensor_buffer, release_tensor_buffer, clear_cache, print_cache_stats

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
    
    if haskey(TENSOR_CACHE, key) && !isempty(TENSOR_CACHE[key])
        buffer = pop!(TENSOR_CACHE[key])
        CACHE_STATS[:hits] += 1
        
        # Asegurarse que tenga la forma correcta (puede ser diferente pero mismo tamaño total)
        return reshape(buffer, shape)
    else
        CACHE_STATS[:misses] += 1
        return CUDA.zeros(type, shape)
    end
end

"""
    release_tensor_buffer(buffer)

Devuelve un buffer al caché para reutilización futura.
"""
function release_tensor_buffer(buffer)
    key = (length(buffer), eltype(buffer))
    
    if !haskey(TENSOR_CACHE, key)
        TENSOR_CACHE[key] = CuArray[]
    end
    
    push!(TENSOR_CACHE[key], buffer)
    CACHE_STATS[:releases] += 1
end

"""
    clear_cache()

Libera toda la memoria almacenada en caché.
"""
function clear_cache()
    empty!(TENSOR_CACHE)
    GC.gc()
    CUDA.reclaim()
    
    # Reiniciar estadísticas
    for key in keys(CACHE_STATS)
        CACHE_STATS[key] = 0
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
    
    println("=== Estadísticas de Caché GPU ===")
    println("Total en caché: $(round(total_bytes/1024^2, digits=2)) MB")
    println("Buffers por tipo:")
    for (type, count) in counts
        println("  $type: $count buffers")
    end
    println("Accesos: $(CACHE_STATS[:hits]) hits, $(CACHE_STATS[:misses]) misses")
    println("Relación hit/miss: $(round(CACHE_STATS[:hits]/(CACHE_STATS[:misses] + 1), digits=2))")
    println("Liberaciones: $(CACHE_STATS[:releases])")
end

end # module