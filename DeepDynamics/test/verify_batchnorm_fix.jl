# verify_batchnorm_fix.jl
# Script para verificar que el fix de collect_parameters funciona correctamente

using DeepDynamics
using Test

println("="^70)
println("VERIFICACIÓN DEL FIX DE BATCHNORM")
println("="^70)

# Test 1: Verificar que collect_parameters incluye BatchNorm
println("\n1️⃣ Test de recolección de parámetros")
model = Sequential([
    Dense(2, 4),
    BatchNorm(4),
    Activation(relu),
    Dense(4, 2)
])

params = collect_parameters(model)
println("  Parámetros totales recolectados: $(length(params))")
println("  Esperados: 8 (2 Dense × 2 + 1 BatchNorm × 2)")

# Verificar específicamente BatchNorm
bn_layer = model.layers[2]
gamma_included = any(p -> p === bn_layer.gamma, params)
beta_included = any(p -> p === bn_layer.beta, params)

println("  BatchNorm gamma incluido: $(gamma_included ? "✅" : "❌")")
println("  BatchNorm beta incluido: $(beta_included ? "✅" : "❌")")

@test length(params) == 8
@test gamma_included
@test beta_included

# Test 2: Verificar convergencia
println("\n2️⃣ Test de convergencia con BatchNorm")

# Datos simples
X = Tensor(Float32[
    1 2 3 4 5 6;
    2 4 6 8 10 12
])

y = Tensor(Float32[
    1 1 1 0 0 0;
    0 0 0 1 1 1
])

# Modelo con BatchNorm
model2 = Sequential([
    Dense(2, 10),
    BatchNorm(10),
    Activation(relu),
    Dense(10, 2),
    Activation(softmax)
])

opt = Adam(learning_rate=0.01f0)
params2 = collect_parameters(model2)

println("  Parámetros del modelo 2: $(length(params2))")

# Entrenar
losses = Float32[]
for epoch in 1:100
    # Zero gradients
    for p in params2
        zero_grad!(p)
    end
    
    # Forward
    pred = model2(X)
    loss = categorical_crossentropy(pred, y)
    push!(losses, loss.data[1])
    
    # Backward
    backward(loss, [1.0f0])
    
    # Update
    optim_step!(opt, params2)
end

initial_loss = losses[1]
final_loss = losses[end]
reduction = (1 - final_loss/initial_loss) * 100

println("  Loss inicial: $(round(initial_loss, digits=4))")
println("  Loss final: $(round(final_loss, digits=4))")
println("  Reducción: $(round(reduction, digits=1))%")

@test final_loss < initial_loss
@test reduction > 30  # Esperamos al menos 30% de mejora

# Verificar que los parámetros de BatchNorm cambiaron
bn2 = model2.layers[2]
gamma_changed = !all(bn2.gamma.data .≈ 1.0f0)
beta_changed = !all(bn2.beta.data .≈ 0.0f0)

println("\n  Parámetros BatchNorm actualizados:")
println("    gamma cambió: $(gamma_changed ? "✅" : "❌")")
println("    beta cambió: $(beta_changed ? "✅" : "❌")")

@test gamma_changed
@test beta_changed

# Test 3: Modelo más complejo con múltiples BatchNorm
println("\n3️⃣ Test con múltiples capas BatchNorm")

model3 = Sequential([
    Dense(10, 20),
    BatchNorm(20),
    Activation(relu),
    DropoutLayer(0.5f0),
    Dense(20, 15),
    BatchNorm(15),
    Activation(relu),
    Dense(15, 5),
    BatchNorm(5),
    Activation(softmax)
])

params3 = collect_parameters(model3)
expected_params = 3 * 2 + 3 * 2  # 3 Dense + 3 BatchNorm

println("  Parámetros recolectados: $(length(params3))")
println("  Esperados: $expected_params")

@test length(params3) == expected_params

# Test 4: Verificar accuracy después del entrenamiento
println("\n4️⃣ Test de accuracy")

# Evaluar modelo
pred_final = model2(X)
pred_classes = argmax(pred_final.data, dims=1)
true_classes = argmax(y.data, dims=1)
accuracy = sum(pred_classes .== true_classes) / length(true_classes)

println("  Accuracy final: $(round(accuracy*100, digits=1))%")
@test accuracy > 0.8  # Esperamos al menos 80% en datos simples

# Test 5: Comparación con modelo sin BatchNorm
println("\n5️⃣ Comparación con/sin BatchNorm")

model_no_bn = Sequential([
    Dense(2, 10),
    Activation(relu),
    Dense(10, 2),
    Activation(softmax)
])

opt_no_bn = Adam(learning_rate=0.01f0)
params_no_bn = collect_parameters(model_no_bn)

losses_no_bn = Float32[]
for epoch in 1:100
    for p in params_no_bn
        zero_grad!(p)
    end
    
    pred = model_no_bn(X)
    loss = categorical_crossentropy(pred, y)
    push!(losses_no_bn, loss.data[1])
    
    backward(loss, [1.0f0])
    optim_step!(opt_no_bn, params_no_bn)
end

final_loss_no_bn = losses_no_bn[end]
println("  Loss final sin BatchNorm: $(round(final_loss_no_bn, digits=4))")
println("  Loss final con BatchNorm: $(round(final_loss, digits=4))")
println("  BatchNorm es $(final_loss < final_loss_no_bn ? "mejor ✅" : "peor ❌")")

println("\n" * "="^70)
println("✨ VERIFICACIÓN COMPLETADA ✨")
println("BatchNorm ahora funciona correctamente en el framework")
println("="^70)

# Resumen de tests
if all([
    length(params) == 8,
    gamma_included,
    beta_included,
    final_loss < initial_loss,
    reduction > 30,
    gamma_changed,
    beta_changed,
    length(params3) == expected_params,
    accuracy > 0.8
])
    println("\n🎉 TODOS LOS TESTS PASARON - EL FIX FUNCIONA CORRECTAMENTE 🎉")
else
    println("\n⚠️  ALGUNOS TESTS FALLARON - REVISAR LA IMPLEMENTACIÓN")
end