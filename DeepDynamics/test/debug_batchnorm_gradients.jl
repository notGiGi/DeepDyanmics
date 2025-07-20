# debug_batchnorm_gradients.jl
# Diagnóstico profundo de por qué BatchNorm no propaga gradientes

using DeepDynamics
using Statistics
using LinearAlgebra
println("="^70)
println("DIAGNÓSTICO PROFUNDO: Gradientes en BatchNorm")
println("="^70)

# Crear un modelo simple con BatchNorm
model = Sequential([
    Dense(2, 4),
    BatchNorm(4),
    Activation(relu),
    Dense(4, 2)
])

# Datos simples
X = Tensor(Float32[1 2; 3 4])
y = Tensor(Float32[1 0; 0 1])

# Obtener parámetros
params = collect_parameters(model)
println("\n1️⃣ Parámetros recolectados: $(length(params))")

# Inicializar gradientes
for p in params
    zero_grad!(p)
end

# Forward pass paso a paso
println("\n2️⃣ Forward pass detallado:")

# Capa 1: Dense
h1 = model.layers[1](X)
println("  Dense output shape: $(size(h1.data))")
println("  Dense output mean: $(mean(h1.data))")

# Capa 2: BatchNorm
h2 = model.layers[2](h1)
println("  BatchNorm output shape: $(size(h2.data))")
println("  BatchNorm output mean: $(mean(h2.data))")
println("  BatchNorm output std: $(std(h2.data))")

# Verificar si BatchNorm tiene backward_fn
println("  BatchNorm output has backward_fn: $(h2.backward_fn !== nothing)")

# Capa 3: ReLU
h3 = model.layers[3](h2)
println("  ReLU output shape: $(size(h3.data))")

# Capa 4: Dense final
h4 = model.layers[4](h3)
println("  Final output shape: $(size(h4.data))")

# Calcular loss
loss = categorical_crossentropy(h4, y)
println("\n3️⃣ Loss: $(loss.data[1])")

# Backward pass
println("\n4️⃣ Backward pass:")
backward(loss, [1.0f0])

# Verificar gradientes
println("\n5️⃣ Gradientes después de backward:")
dense1 = model.layers[1]
bn = model.layers[2]
dense2 = model.layers[4]

println("\n  Dense 1:")
println("    weights grad exists: $(dense1.weights.grad !== nothing)")
if dense1.weights.grad !== nothing
    println("    weights grad norm: $(norm(dense1.weights.grad.data))")
    println("    weights grad mean: $(mean(dense1.weights.grad.data))")
end

println("\n  BatchNorm:")
println("    gamma grad exists: $(bn.gamma.grad !== nothing)")
if bn.gamma.grad !== nothing
    println("    gamma grad norm: $(norm(bn.gamma.grad.data))")
    println("    gamma grad mean: $(mean(bn.gamma.grad.data))")
    println("    gamma grad values: $(bn.gamma.grad.data)")
end
println("    beta grad exists: $(bn.beta.grad !== nothing)")
if bn.beta.grad !== nothing
    println("    beta grad norm: $(norm(bn.beta.grad.data))")
    println("    beta grad mean: $(mean(bn.beta.grad.data))")
    println("    beta grad values: $(bn.beta.grad.data)")
end

println("\n  Dense 2:")
println("    weights grad exists: $(dense2.weights.grad !== nothing)")
if dense2.weights.grad !== nothing
    println("    weights grad norm: $(norm(dense2.weights.grad.data))")
end

# Test simplificado: Solo BatchNorm
println("\n" * "="^70)
println("6️⃣ Test aislado de BatchNorm:")
println("="^70)

# Crear BatchNorm aislado
bn_test = BatchNorm(2, training=true)

# Entrada simple
x_test = Tensor(Float32[1 2; 3 4]; requires_grad=true)
zero_grad!(x_test)
zero_grad!(bn_test.gamma)
zero_grad!(bn_test.beta)

# Forward
y_test = bn_test(x_test)
println("  Input: $(x_test.data)")
println("  Output: $(y_test.data)")
println("  Output has backward_fn: $(y_test.backward_fn !== nothing)")

# Loss simple (suma)
loss_test = sum(y_test.data)
loss_tensor = Tensor([loss_test]; requires_grad=true)

# Definir backward manual
loss_tensor.backward_fn = _ -> begin
    if y_test.backward_fn !== nothing
        y_test.backward_fn(ones(size(y_test.data)))
    else
        println("  ⚠️ y_test NO tiene backward_fn!")
    end
end

# Backward
backward(loss_tensor, [1.0f0])

println("\n  Gradientes después de backward:")
println("    x_test grad: $(x_test.grad !== nothing ? x_test.grad.data : "nothing")")
println("    gamma grad: $(bn_test.gamma.grad !== nothing ? bn_test.gamma.grad.data : "nothing")")
println("    beta grad: $(bn_test.beta.grad !== nothing ? bn_test.beta.grad.data : "nothing")")

# Verificar si el problema es que BatchNorm no define backward_fn
println("\n7️⃣ Análisis del problema:")
if y_test.backward_fn === nothing
    println("  ❌ PROBLEMA ENCONTRADO: BatchNorm no define backward_fn!")
    println("     Esto explica por qué los gradientes no se propagan.")
else
    println("  ✅ BatchNorm sí define backward_fn")
end

println("\n" * "="^70)