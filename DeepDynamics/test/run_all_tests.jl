println("\n🚀 EJECUTANDO TODOS LOS TESTS (FASES 1-4)\n")

let
    total_passed = 0
    total_failed = 0
    failed_tests = String[]

    # Lista de todos los tests
    tests = [
        # FASE 1
        ("src/test_phase1.jl", "Test Básico FASE 1"),
        ("test/test_regression_fase1.jl", "Test de Regresión FASE 1"),
        ("test/test_gradient_accumulation.jl", "Test de Acumulación de Gradientes"),
        ("test/test_requires_grad.jl", "Test de requires_grad"),
        ("test/test_training_complete.jl", "Test de Entrenamiento Completo"),

        # FASE 2
        ("src/test_phase2.jl", "Test Básico FASE 2"),
        ("test/test_integration_phase1_2.jl", "Test Integración FASE 1+2"),

        # FASE 3
        ("test/test_phase3.jl", "Test Capas Fundamentales FASE 3"),
        ("test/test_phase3_integration.jl", "Test Integración FASE 3"),
        ("test/integration_test.jl", "Test Integración CNN FASE 3"),
        ("test/test_regresion_lineal.jl", "Test Regresión Lineal FASE 3"),

        # FASE 4
        ("test/test_phase4.jl", "Test Fase 4 - BatchNorm"),
        ("test/diagnose_batchnorm.jl", "Diagnóstico BatchNorm FASE 4"),
        ("test/quick_verify_batchnorm.jl", "Verificación Rápida BatchNorm"),
        ("test/test_phase4_full.jl", "Fase 4 Full"),

        #FASE 5
        ("test/test_phase5.jl", "Test Fase 5 - CNN"),

        #FASE &
        ("test/test_phase6.jl", "Test Fase 6 - GPU")
    ]

    for (file, name) in tests
        println("\n" * "="^60)
        println("📋 $name")
        println("="^60)
        try
            include(joinpath(@__DIR__, "..", file))
            total_passed += 1
            println("\n✅ $name PASÓ")
        catch e
            total_failed += 1
            push!(failed_tests, name)
            println("\n❌ $name FALLÓ:")
            println(e)
        end
    end

    # Resumen final
    println("\n" * "="^60)
    println("📊 RESUMEN FINAL:")
    println("   ✅ Pasaron: $total_passed")
    println("   ❌ Fallaron: $total_failed")
    if !isempty(failed_tests)
        println("\nTests fallidos:")
        for t in failed_tests
            println("   - $t")
        end
    end
    println("="^60)
end
