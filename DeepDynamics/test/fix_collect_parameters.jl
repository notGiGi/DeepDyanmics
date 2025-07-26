# fix_collect_parameters.jl
# Verificar y arreglar la función collect_parameters

using DeepDynamics
using DeepDynamics.TensorEngine
println("="^60)
println("DIAGNÓSTICO: collect_parameters")
println("="^60)

# Crear modelo con BatchNorm
model = Sequential([
    Dense(2, 4),
    BatchNorm(4),
    Activation(relu),
    Dense(4, 2)
])

# Verificar qué está recolectando collect_parameters
params = collect_parameters(model)

println("\nParámetros recolectados:")
println("  Total: ", length(params))

# Verificar manualmente qué debería haber
expected_params = 0
for layer in model.layers
    if layer isa Dense
        expected_params += 2  # weights + biases
    elseif layer isa BatchNorm
        expected_params += 2  # gamma + beta
    end
end

println("  Esperados: ", expected_params)
println("  ¿Coinciden? ", length(params) == expected_params ? "✅" : "❌")

# Verificar si BatchNorm está incluido
bn_layer = model.layers[2]
gamma_included = any(p -> p === bn_layer.gamma, params)
beta_included = any(p -> p === bn_layer.beta, params)

println("\nBatchNorm en collect_parameters:")
println("  gamma incluido: ", gamma_included ? "✅" : "❌")
println("  beta incluido: ", beta_included ? "✅" : "❌")

# Si no están incluidos, necesitamos arreglar collect_parameters
if !gamma_included || !beta_included
    println("\n⚠️  PROBLEMA DETECTADO: collect_parameters no incluye BatchNorm")
    println("\nRevisando implementación actual...")
    
    # Aquí está el fix para NeuralNetwork.jl
    println("""
    
    SOLUCIÓN: Agregar el siguiente código en NeuralNetwork.jl:
    
    function collect_parameters(model::Sequential)
        params = TensorEngine.Tensor[]
        for layer in model.layers
            if layer isa Dense
                push!(params, layer.weights, layer.biases)
            elseif layer isa BatchNorm
                push!(params, layer.gamma, layer.beta)  # <-- AGREGAR ESTO
            elseif layer isa ConvKernelLayers.ConvKernelLayer
                push!(params, TensorEngine.Tensor(layer.weights), TensorEngine.Tensor(layer.bias))
            end
        end
        return params
    end
    """)
end

# Test rápido con el fix simulado
println("\n" * "="^60)
println("TEST: Convergencia con fix simulado")
println("="^60)

# Función temporal que recolecta parámetros correctamente
function collect_parameters_fixed(model)
    params = TensorEngine.Tensor[]
    for layer in model.layers
        if layer isa Dense
            push!(params, layer.weights, layer.biases)
        elseif layer isa BatchNorm
            push!(params, layer.gamma, layer.beta)  # FIX
        end
    end
    return params
end

# Datos simples
X = Tensor(Float32[1 2 3 4; 2 4 6 8])
y = Tensor(Float32[0 0 1 1; 1 1 0 0])

# Modelo con BatchNorm
model2 = Sequential([
    Dense(2, 4),
    BatchNorm(4),
    Activation(relu),
    Dense(4, 2),
    Activation(softmax)
])

opt = Adam(learning_rate=0.01f0)
params_fixed = collect_parameters_fixed(model2)

println("\nParámetros con fix: ", length(params_fixed))

losses = Float32[]
for i in 1:100
    for p in params_fixed
        zero_grad!(p)
    end
    
    pred = model2(X)
    loss = categorical_crossentropy(pred, y)
    push!(losses, loss.data[1])
    
    backward(loss, [1.0f0])
    optim_step!(opt, params_fixed)
end

println("\nResultados con fix:")
println("  Loss inicial: ", losses[1])
println("  Loss final: ", losses[end])
println("  Reducción: ", round((1 - losses[end]/losses[1])*100, digits=1), "%")
println("  ¿Converge? ", losses[end] < losses[1] ? "✅" : "❌")

# Verificar que los parámetros de BatchNorm cambiaron
bn = model2.layers[2]
println("\nCambios en BatchNorm:")
println("  gamma[1] inicial ≈ 1.0, final = ", bn.gamma.data[1])
println("  beta[1] inicial ≈ 0.0, final = ", bn.beta.data[1])

println("\n" * "="^60)