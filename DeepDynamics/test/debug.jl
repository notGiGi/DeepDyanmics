module DeepDebug
# Debug de shapes/layers para DeepDynamics (solo forward)

using DeepDynamics
const TE = DeepDynamics.TensorEngine
const NN = DeepDynamics.NeuralNetwork

# ------------------ helpers compactos ------------------
safe_data(x) = x isa TE.Tensor ? x.data : x
safe_size(x) = try size(safe_data(x)) catch; () end
typename(x)  = x isa Type ? string(x) : string(typeof(x))
shortshape(x)= try "(" * join(safe_size(x), ",") * ")" catch; "<no-size>" end

# chequeo rápido para RNN/LSTM/GRU
function check_sequence_batch(X; name="X")
    A = safe_data(X); nd = ndims(A)
    if nd == 3
        (F,B,T) = size(A)
        println("[$name] 3D ok (F,B,T)=($F,$B,$T)")
        return true
    elseif nd == 4
        (F,B,D3,D4) = size(A)
        ok = (D3==1 || D4==1)
        println("[$name] 4D (F,B,$D3,$D4)  ", ok ? "✓ singleton ok" : "✗ falta singleton")
        return ok
    else
        println("[$name] ndims=$nd (esperado 3D o 4D con singleton)")
        return false
    end
end

# ------------------ trazador por capa ------------------
"""
    trace_forward(model::Sequential, x; stop_on_error=true)

Imprime capa por capa: tipo, shape in -> out.
Si hay error, muestra contexto y pistas típicas para RNN/LSTM/GRU.
"""
function trace_forward(model::DeepDynamics.Sequential, x; stop_on_error::Bool=true)
    x_t = x isa TE.Tensor ? x : TE.Tensor(x)
    println("—"^70)
    println("TRACE model=Sequential[", length(model.layers), " layers]  input=", shortshape(x_t))
    cur = x_t
    for (i, layer) in enumerate(model.layers)
        print(@sprintf("[%02d] %-24s  in=%-16s  ", i, typename(layer), shortshape(cur)))
        try
            out = NN.forward(layer, cur)
            println("out=", shortshape(out))
            cur = out
        catch e
            println("❌ ERROR")
            println("\n>>> ERROR en capa [$i] ", typename(layer))
            println("    input shape: ", shortshape(cur))
            if e isa DimensionMismatch
                println("    DimensionMismatch: ", e)
                println("    Tip: RNN/LSTM/GRU en este repo esperan (F,B,T).")
                println("         Si tienes (B,F,T) o (F,T,B), permuta ejes antes de la primera RNN.")
                println("         Para 4D, aceptan (F,B,1,T) o (F,B,T,1) (una dim unitaria).")
            elseif e isa MethodError
                println("    MethodError: ", e)
                println("    Tip: usa NN.forward(layer, x) (Sequential ya lo hace).")
            else
                println("    ", typeof(e), ": ", e)
            end
            # stack corto
            bt = stacktrace(e)
            println("    Stack (top 5):")
            for (k, fr) in enumerate(bt[1:min(5,length(bt))])
                println("      $k) ", fr)
            end
            stop_on_error && rethrow(e)
        end
    end
    println("—"^70)
    return cur
end

# ------------------ runner sencillo ------------------
"""
    run(model, X; name="X")

Chequea dims para secuencias y ejecuta trace_forward (no lanza excepción).
"""
function run(model::DeepDynamics.Sequential, X; name::String="X")
    println("== DeepDebug ==")
    println("batch ", name, " shape: ", shortshape(X))
    check_sequence_batch(X; name=name)
    return trace_forward(model, X; stop_on_error=false)
end

end # module
