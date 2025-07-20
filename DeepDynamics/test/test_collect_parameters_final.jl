# test_collect_parameters_final.jl
# Test exhaustivo y solución para collect_parameters

using DeepDynamics
using DeepDynamics.TensorEngine
println("="^70)
println("TEST EXHAUSTIVO: collect_parameters")
println("="^70)

# Crear modelo de prueba
model = Sequential([
    Dense(2, 4),
    BatchNorm(4), 
    Activation(relu),
    Dense(4, 2)
])

# Test 1: Verificar la función actual
println("\n1️⃣ Probando collect_parameters actual:")
params_original = collect_parameters(model)
println("  Parámetros recolectados: $(length(params_original))")

# Test 2: Implementación manual directa
println("\n2️⃣ Recolección manual directa:")
params_manual = TensorEngine.Tensor[]

# Dense 1
push!(params_manual, model.layers[1].weights)
push!(params_manual, model.layers[1].biases)
println("  Dense 1: 2 parámetros agregados")

# BatchNorm
push!(params_manual, model.layers[2].gamma)
push!(params_manual, model.layers[2].beta)
println("  BatchNorm: 2 parámetros agregados")

# Activation - sin parámetros
println("  Activation: 0 parámetros")

# Dense 2
push!(params_manual, model.layers[4].weights)
push!(params_manual, model.layers[4].biases)
println("  Dense 2: 2 parámetros agregados")

println("  Total manual: $(length(params_manual))")

# Test 3: Función collect_parameters mejorada localmente
function collect_parameters_fixed(model::Sequential)
    params = TensorEngine.Tensor[]
    
    for (i, layer) in enumerate(model.layers)
        # Imprimir qué estamos procesando
        layer_type = split(string(typeof(layer)), ".")[end]
        print("  Procesando capa $i ($layer_type): ")
        
        if isa(layer, DeepDynamics.NeuralNetwork.Dense)
            push!(params, layer.weights)
            push!(params, layer.biases)
            println("2 params agregados (Dense)")
            
        elseif isa(layer, DeepDynamics.Layers.BatchNorm)
            push!(params, layer.gamma)
            push!(params, layer.beta)
            println("2 params agregados (BatchNorm)")
            
        elseif isa(layer, DeepDynamics.ConvKernelLayers.ConvKernelLayer)
            if layer.weights isa TensorEngine.Tensor
                push!(params, layer.weights)
            else
                push!(params, TensorEngine.Tensor(layer.weights))
            end
            
            if layer.bias isa TensorEngine.Tensor
                push!(params, layer.bias)
            else
                push!(params, TensorEngine.Tensor(layer.bias))
            end
            println("2 params agregados (ConvKernel)")
            
        else
            println("sin parámetros")
        end
    end
    
    return params
end

println("\n3️⃣ Probando collect_parameters_fixed:")
params_fixed = collect_parameters_fixed(model)
println("  Total con fix: $(length(params_fixed))")

# Test 4: Verificar convergencia con diferentes conjuntos de parámetros
println("\n4️⃣ Test de convergencia:")

X = Tensor(Float32[1 2 3 4 5 6; 2 4 6 8 10 12])
y = Tensor(Float32[1 1 1 0 0 0; 0 0 0 1 1 1])

# Con parámetros originales
if length(params_original) > 0
    println("\n  a) Con params originales ($(length(params_original)) params):")
    opt1 = Adam(learning_rate=0.01f0)
    
    losses1 = Float32[]
    for epoch in 1:50
        for p in params_original
            zero_grad!(p)
        end
        
        pred = model(X)
        loss = categorical_crossentropy(pred, y)
        push!(losses1, loss.data[1])
        
        backward(loss, [1.0f0])
        optim_step!(opt1, params_original)
    end
    
    println("    Loss inicial: $(losses1[1])")
    println("    Loss final: $(losses1[end])")
    println("    Reducción: $(round((1 - losses1[end]/losses1[1])*100, digits=1))%")
end

# Con parámetros manuales
println("\n  b) Con params manuales ($(length(params_manual)) params):")
opt2 = Adam(learning_rate=0.01f0)

# Resetear modelo
model2 = Sequential([
    Dense(2, 4),
    BatchNorm(4),
    Activation(relu),
    Dense(4, 2)
])

# Recolectar params manualmente de nuevo
params_manual2 = TensorEngine.Tensor[]
push!(params_manual2, model2.layers[1].weights)
push!(params_manual2, model2.layers[1].biases)
push!(params_manual2, model2.layers[2].gamma)
push!(params_manual2, model2.layers[2].beta)
push!(params_manual2, model2.layers[4].weights)
push!(params_manual2, model2.layers[4].biases)

losses2 = Float32[]
for epoch in 1:50
    for p in params_manual2
        zero_grad!(p)
    end
    
    pred = model2(X)
    loss = categorical_crossentropy(pred, y)
    push!(losses2, loss.data[1])
    
    backward(loss, [1.0f0])
    optim_step!(opt2, params_manual2)
end

println("    Loss inicial: $(losses2[1])")
println("    Loss final: $(losses2[end])")
println("    Reducción: $(round((1 - losses2[end]/losses2[1])*100, digits=1))%")

# Test 5: Diagnóstico del problema
println("\n5️⃣ Diagnóstico del problema:")

# Verificar si el problema es el tipo
bn = model.layers[2]
println("  typeof(bn): $(typeof(bn))")
println("  BatchNorm fullname: $(typeof(DeepDynamics.Layers.BatchNorm))")
println("  ¿Son el mismo tipo?: $(typeof(bn) === DeepDynamics.Layers.BatchNorm)")

# Test con comparación directa
for (i, layer) in enumerate(model.layers)
    result = if isa(layer, DeepDynamics.NeuralNetwork.Dense)
        "Dense ✓"
    elseif isa(layer, DeepDynamics.Layers.BatchNorm)
        "BatchNorm ✓"
    elseif isa(layer, DeepDynamics.NeuralNetwork.Activation)
        "Activation ✓"
    else
        "Desconocido ✗"
    end
    println("  Capa $i: $result")
end

# SOLUCIÓN PROPUESTA
println("\n" * "="^70)
println("💡 SOLUCIÓN PROPUESTA PARA NeuralNetwork.jl:")
println("="^70)

println("""
La función collect_parameters debe verificar tipos con namespace completo.
Aquí está el código corregido que debe reemplazar la función actual:

```julia
function collect_parameters(model::Sequential)
    params = TensorEngine.Tensor[]
    
    for layer in model.layers
        if isa(layer, Dense)  # Este funciona porque está en el mismo módulo
            push!(params, layer.weights)
            push!(params, layer.biases)
            
        elseif isa(layer, Layers.BatchNorm)  # Namespace completo
            push!(params, layer.gamma)
            push!(params, layer.beta)
            
        elseif isa(layer, ConvKernelLayers.ConvKernelLayer)
            if layer.weights isa TensorEngine.Tensor
                push!(params, layer.weights)
            else
                push!(params, TensorEngine.Tensor(layer.weights))
            end
            
            if layer.bias isa TensorEngine.Tensor
                push!(params, layer.bias)
            else
                push!(params, TensorEngine.Tensor(layer.bias))
            end
            
        elseif isa(layer, ConvolutionalLayers.Conv2D)
            push!(params, layer.weights)
            push!(params, layer.bias)
            
            if layer.use_batchnorm && layer.gamma !== nothing
                push!(params, layer.gamma)
                push!(params, layer.beta)
            end
            
        elseif isa(layer, Layers.ResidualBlock)
            for sublayer in layer.conv_path
                append!(params, collect_parameters(Sequential([sublayer])))
            end
            for sublayer in layer.shortcut
                append!(params, collect_parameters(Sequential([sublayer])))
            end
        end
    end
    
    return params
end
```

IMPORTANTE: El problema es que la comparación con BatchNorm debe usar 
el namespace completo `Layers.BatchNorm` para funcionar correctamente.
""")

println("\n" * "="^70)