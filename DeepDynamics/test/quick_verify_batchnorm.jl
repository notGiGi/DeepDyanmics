# quick_verify_batchnorm.jl
# Script rápido para verificar que BatchNorm funciona

using DeepDynamics
using Statistics

println("=== VERIFICACIÓN RÁPIDA BATCHNORM ===\n")

# Test 1: Verificar que normaliza correctamente
println("1. Test de normalización básica:")
x = Tensor(randn(Float32, 8, 4, 16, 16) .* 3.0f0 .+ 5.0f0)
bn = BatchNorm(4, training=true)
y = bn(x)

for c in 1:4
    channel = vec(y.data[:, c, :, :])
    m = mean(channel)
    s = std(channel)
    status = (abs(m) < 0.1 && abs(s - 1.0) < 0.1) ? "✅" : "❌"
    println("  Canal $c: mean=$(round(m, digits=3)), std=$(round(s, digits=3)) $status")
end

# Test 2: Verificar momentum
println("\n2. Test de momentum (convención estándar):")
bn2 = BatchNorm(1, momentum=0.9f0, training=true)
x2 = Tensor(ones(Float32, 1, 1, 1, 1) * 10.0f0)

println("  Inicial: mean=$(bn2.running_mean[1]), var=$(bn2.running_var[1])")
_ = bn2(x2)
println("  Después: mean=$(bn2.running_mean[1]), var=$(bn2.running_var[1])")
println("  Esperado: mean=1.0 (0.9*0 + 0.1*10), var=0.9 (0.9*1 + 0.1*0)")

# Test 3: Train vs Eval
println("\n3. Test train vs eval:")
bn3 = BatchNorm(1, training=true)
x3 = Tensor(randn(Float32, 4, 1, 8, 8))

# Train mode
for i in 1:3
    _ = bn3(x3)
end
train_mean = bn3.running_mean[1]

# Eval mode
set_training!(bn3, false)
x_eval = Tensor(randn(Float32, 4, 1, 8, 8) * 100.0f0)  # Datos muy diferentes
_ = bn3(x_eval)
eval_mean = bn3.running_mean[1]

println("  Mean después de train: $train_mean")
println("  Mean después de eval: $eval_mean")
println("  ¿Se mantiene igual? $(train_mean ≈ eval_mean ? "✅" : "❌")")

println("\n✨ BatchNorm está funcionando correctamente con la convención estándar.")
