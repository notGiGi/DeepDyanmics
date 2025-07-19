println("\n🚀 EJECUTANDO TODOS LOS TESTS (FASE 1 + FASE 2)\n")

tests = [
    # FASE 1
    ("src/test_phase1.jl", "Test Básico FASE 1"),
    ("test/test_regression_fase1.jl", "Test de Regresión FASE 1"),
    ("test/test_gradient_accumulation.jl", "Test de Acumulación de Gradientes"),
    ("test/test_requires_grad.jl", "Test de requires_grad"),
    ("test/test_training_complete.jl", "Test de Entrenamiento Completo"),
    
    # FASE 2
    ("src/test_phase2.jl", "Test Básico FASE 2"),
    ("test/test_integration_phase1_2.jl", "Test Integración FASE 1+2")
]

passed = 0
failed = 0
failed_tests = String[]

for (file, name) in tests
    println("\n" * "="^60)
    println("📋 $name")
    println("="^60)
    
    try
        include(file)
        passed += 1
        println("\n✅ $name PASÓ")
    catch e
        failed += 1
        push!(failed_tests, name)
        println("\n❌ $name FALLÓ:")
        println(e)
    end
end

println("\n" * "="^60)
println("📊 RESUMEN FINAL:")
println("   ✅ Pasaron: $passed")
println("   ❌ Fallaron: $failed")
if !isempty(failed_tests)
    println("\nTests fallidos:")
    for test in failed_tests
        println("   - $test")
            end
end
println("="^60)
