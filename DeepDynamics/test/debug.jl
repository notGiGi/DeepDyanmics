# SESIÓN DE DEBUGGING REAL - FASE 7
# Ejecutar línea por línea en el REPL

using DeepDynamics
using Test
using DeepDynamics
using CUDA
using DeepDynamics.NeuralNetwork
using  DeepDynamics.TensorEngine
using DeepDynamics.Losses  # o import DeepDynamics.Losses
using Statistics
using DeepDynamics.Optimizers
using LinearAlgebra
# 1. CREAR MODELO SIMPLE PARA DEBUGGING
model = Sequential([
    Dense(5, 3),
    Activation(softmax)
])

# 2. DATOS DE PRUEBA SIMPLES
X = Tensor(randn(Float32, 5, 2))  # 2 muestras, 5 features
y_true = Tensor(Float32[1 0; 0 1; 0 0])  # one-hot encoding

# 3. VERIFICAR FORWARD PASS
output = forward(model, X)

println("Output shape: ", size(output.data))
println("Output values:\n", output.data)
println("¿Suman a 1.0? ", sum(output.data, dims=1))  # Debe ser [1.0, 1.0]

# 4. CALCULAR LOSS
loss = categorical_crossentropy(output, y_true)
println("Loss value: ", loss.data[1])

# 5. VERIFICAR GRADIENTES ANTES DE BACKWARD
params = collect_parameters(model)
println("Parámetros encontrados: ", length(params))

for p in params
    println("Grad antes: ", p.grad === nothing ? "nothing" : "exists")
end

# 6. BACKWARD
TensorEngine.backward(loss, [1.0f0])

# 7. INSPECCIONAR GRADIENTES
for (i, p) in enumerate(params)
    if p.grad !== nothing
        println("\nParam $i:")
        println("  Shape: ", size(p.grad.data))
        println("  Norm: ", norm(p.grad.data))
        println("  Max: ", maximum(abs.(p.grad.data)))
        println("  ¿Tiene NaN?: ", any(isnan.(p.grad.data)))
        println("  ¿Tiene Inf?: ", any(isinf.(p.grad.data)))
    end
end

# 8. TEST ESPECÍFICO: SOFTMAX + CROSSENTROPY
println("\n=== TEST GRADIENTE MANUAL ===")

# Crear entrada simple
logits = Tensor(Float32[2.0 1.0; 1.0 2.0; 0.0 0.0])  # 3 clases, 2 muestras
probs = softmax(logits)
println("Probabilidades después de softmax:\n", probs.data)

# Verificar que es una distribución válida
println("Sumas por columna: ", sum(probs.data, dims=1))

# Target one-hot
target = Tensor(Float32[1 0; 0 1; 0 0])

# Loss
loss_manual = categorical_crossentropy(probs, target)
println("\nLoss manual: ", loss_manual.data[1])

# Backward
TensorEngine.zero_grad!(logits)
TensorEngine.backward(loss_manual, [1.0f0])

# Verificar gradiente
if logits.grad !== nothing
    println("\nGradiente en logits:")
    println(logits.grad.data)
    
    # Gradiente esperado: (probs - target) / batch_size
    expected_grad = (probs.data .- target.data) ./ 2.0f0
    println("\nGradiente esperado:")
    println(expected_grad)
    
    println("\n¿Son iguales? ", isapprox(logits.grad.data, expected_grad, rtol=1e-4))
end

# 9. TEST CON BATCHNORM
println("\n=== TEST CON BATCHNORM ===")

model_bn = Sequential([
    Dense(5, 10),
    BatchNorm(10),
    Activation(relu),
    Dense(10, 3),
    Activation(softmax)
])

X_bn = Tensor(randn(Float32, 5, 8))  # 8 muestras
y_bn = Tensor(zeros(Float32, 3, 8))
for i in 1:8
    y_bn.data[rand(1:3), i] = 1.0f0
end

# Inicializar parámetros
params_bn = collect_parameters(model_bn)
for p in params_bn
    TensorEngine.zero_grad!(p)
end

# Forward
out_bn = forward(model_bn, X_bn)
println("Output con BN - shape: ", size(out_bn.data))
println("Output con BN - sumas: ", sum(out_bn.data, dims=1))

# Loss
loss_bn = categorical_crossentropy(out_bn, y_bn)
println("Loss inicial: ", loss_bn.data[1])

# Backward
TensorEngine.backward(loss_bn, [1.0f0])

# Verificar gradientes
for (i, p) in enumerate(params_bn)
    if p.grad !== nothing
        grad_norm = norm(p.grad.data)
        println("Param $i - grad norm: ", grad_norm)
        if grad_norm > 10.0
            println("  ⚠️ GRADIENTE GRANDE!")
        end
    end
end

# 10. SIMULAR ENTRENAMIENTO
println("\n=== SIMULACIÓN DE ENTRENAMIENTO ===")

opt = SGD(learning_rate=0.01)
losses = Float32[]

for epoch in 1:10
    # Zero grad
    for p in params_bn
        TensorEngine.zero_grad!(p)
    end
    
    # Forward
    out = forward(model_bn, X_bn)
    loss = categorical_crossentropy(out, y_bn)
    push!(losses, loss.data[1])
    
    # Backward
    TensorEngine.backward(loss, [1.0f0])
    
    # Check for gradient explosion
    max_grad = maximum([norm(p.grad.data) for p in params_bn if p.grad !== nothing])
    
    # Update
    Optimizers.step!(opt, params_bn)
    
    println("Epoch $epoch: loss = $(loss.data[1]), max_grad_norm = $max_grad")
end

println("\n¿Loss disminuye? ", losses[end] < losses[1])
println("Cambio: ", losses[1], " -> ", losses[end])

# 11. DIAGNÓSTICO FINAL
if losses[end] > losses[1]
    println("\n🚨 PROBLEMA DETECTADO: Loss aumenta!")
    println("\nPosibles causas:")
    println("1. Learning rate muy alto")
    println("2. Gradientes mal calculados")
    println("3. Problema en BatchNorm")
    
    # Test específico
    println("\n=== TEST SIN BATCHNORM ===")
    model_simple = Sequential([
        Dense(5, 10),
        Activation(relu),
        Dense(10, 3),
        Activation(softmax)
    ])
    
    params_simple = collect_parameters(model_simple)
    losses_simple = Float32[]
    
    for epoch in 1:10
        for p in params_simple
            TensorEngine.zero_grad!(p)
        end
        
        out = forward(model_simple, X_bn)
        loss = categorical_crossentropy(out, y_bn)
        push!(losses_simple, loss.data[1])
        
        TensorEngine.backward(loss, [1.0f0])
        Optimizers.step!(opt, params_simple)
    end
    
    println("\nSin BatchNorm: ", losses_simple[1], " -> ", losses_simple[end])
    println("¿Mejora sin BatchNorm? ", losses_simple[end] < losses_simple[1])
end