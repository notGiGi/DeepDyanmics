module ProfilingTools

using BenchmarkTools
using Profile
export @profile_block, print_profile_summary

"Macro para medir el tiempo de ejecución de un bloque de código usando BenchmarkTools."
macro profile_block(name, expr)
    return quote
        println("Iniciando bloque de profiling: ", $(esc(name)))
        t = @belapsed $(esc(expr))
        println("Tiempo transcurrido en $(string($(esc(name)))): ", t, " segundos")
        $(esc(expr))
    end
end

"Imprime un resumen del perfil de ejecución obtenido con el profiler estándar de Julia."
function print_profile_summary()
    try
        
        println("Resumen del perfil:")
        Profile.print()
    catch err
        println("No se pudo imprimir el perfil: ", err)
    end
end

end  # module ProfilingTools
