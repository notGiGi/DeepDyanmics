module GPUUtils

using CUDA

export safe_memory_info, gpu_available, clear_gpu_memory

"""
    safe_memory_info()

Obtiene información de memoria GPU de manera segura, manejando casos donde CUDA.memory_status() falla.
"""
function safe_memory_info()
    if !CUDA.functional()
        return nothing
    end
    
    try
        mem_info = CUDA.memory_status()
        return mem_info
    catch e
        println("No se pudo obtener información de memoria: $e")
        return nothing
    end
end

"""
    gpu_available()

Comprueba si hay GPU disponible para cálculos.
"""
function gpu_available()
    return CUDA.functional()
end

"""
    clear_gpu_memory()

Libera memoria GPU.
"""
function clear_gpu_memory()
    if CUDA.functional()
        GC.gc()
        CUDA.reclaim()
        return true
    end
    return false
end

"""
    print_gpu_info()

Imprime información detallada sobre la GPU disponible.
"""
function print_gpu_info()
    if !CUDA.functional()
        println("GPU no disponible")
        return
    end
    
    println("GPU disponible: ", CUDA.name(CUDA.device()))
    
    mem_info = safe_memory_info()
    if mem_info !== nothing && length(mem_info) >= 2
        println("Memoria GPU: $(round(mem_info[1]/1024^3, digits=2)) GB libre de $(round(mem_info[2]/1024^3, digits=2)) GB total")
    else
        println("Información de memoria GPU no disponible")
    end
end

end # module