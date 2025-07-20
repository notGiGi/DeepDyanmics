# diagnose_batchnorm.jl
# Script para diagnosticar el problema con BatchNorm

using DeepDynamics
using Statistics

println("=== DIAGNÓSTICO DE BATCHNORM ===\n")

# Test 1: Verificar la fórmula de momentum
println("1. Verificando fórmula de momentum:")

x_data = zeros(Float32, 2, 3, 4, 4)
for c in 1:3 x_data[:, c, :, :] .= Float32(c - 1) end
bn = BatchNorm(3, training=true, momentum=0.1f0)
println("Estado inicial:")
println("  running_mean = ", bn.running_mean)
println("  running_var = ", bn.running_var)

x = Tensor(x_data)
y = bn(x)

println("\nDespués del forward:")
println("  running_mean = ", bn.running_mean)
println("  running_var = ", bn.running_var)

batch_mean = [0.0f0, 1.0f0, 2.0f0]
batch_var = [0.0f0, 0.0f0, 0.0f0]
println("\nEstadísticas del batch:")
println("  batch_mean = ", batch_mean)
println("  batch_var = ", batch_var)

println("\nInterpretación 1 (PyTorch style - momentum=0.1):")
expected_mean_1 = 0.1f0 .* [0.0, 0.0, 0.0] .+ 0.9f0 .* batch_mean
expected_var_1 = 0.1f0 .* [1.0, 1.0, 1.0] .+ 0.9f0 .* batch_var
println("  expected_mean = ", expected_mean_1)
println("  expected_var = ", expected_var_1)

println("\nInterpretación 2 (Test expectation):")
expected_mean_2 = 0.9f0 .* [0.0, 0.0, 0.0] .+ 0.1f0 .* batch_mean
expected_var_2 = 0.9f0 .* [1.0, 1.0, 1.0] .+ 0.1f0 .* batch_var
println("  expected_mean = ", expected_mean_2)
println("  expected_var = ", expected_var_2)

println("\n¿Cuál coincide con lo obtenido?")
if all(abs.(bn.running_mean .- expected_mean_1) .< 0.001)
    println("  ✓ Coincide con Interpretación 1 (PyTorch style)")
elseif all(abs.(bn.running_mean .- expected_mean_2) .< 0.001)
    println("  ✓ Coincide con Interpretación 2 (Test expectation)")
else
    println("  ✗ No coincide con ninguna interpretación")
end
