    using DeepDynamics
    using Test

println("=== TEST FASE 2: Optimizadores Mejorados ===")

@testset "FASE 2 - Adam y optimizaciones" begin
    # Verificar creación de Adam
    opt = Adam(learning_rate=0.001, weight_decay=0.0001)
        @test opt.learning_rate == 0.001f0
    @test opt.weight_decay == 0.0001f0

    # Test dummy con Tensor
    p = Tensor(randn(Float32, 5, 5); requires_grad=true)
        p.grad = Tensor(randn(Float32, 5, 5))
    old_p = copy(p.data)
    
        # Aplicar un paso
    DeepDynamics.optim_step!(opt, [p])

    # Verificar que los parámetros cambiaron
    @test !isapprox(old_p, p.data)
end

println("✓ TEST FASE 2 COMPLETADO CON ÉXITO")
