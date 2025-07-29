using DeepDynamics
using Test
using DeepDynamics: Activation
println("\n=== TEST INTEGRACIÓN FASE 3 (CORREGIDO) ===")

@testset "Integración Fase 3" begin
    # Modelo simple sin capas problemáticas
    model = Sequential([
        Dense(10, 20),
        Activation(relu),
        Dense(20, 5)
    ])

    
    # Datos de entrada
    X = Tensor(randn(Float32, 10, 4))  # 10 features, 4 samples
    y_true = Tensor(randn(Float32, 5, 4))  # 5 outputs, 4 samples
    
    # Forward
    y_pred = model(X)
    @test size(y_pred.data) == (5, 4)
    
    # Loss
    loss = mse_loss(y_pred, y_true)
    @test size(loss.data) == (1,)
    
    # Backward
    params = collect_parameters(model)
    for p in params
        zero_grad!(p)
    end
    
    backward(loss, [1.0f0])
    
    # Verificar gradientes
    for p in params
        @test p.grad !== nothing
        @test !any(isnan.(p.grad.data))
    end
    
    println("✓ Integración básica funciona")
    
    # Test con Flatten
    model2 = Sequential([
        Flatten(),
        Dense(3*32*32, 10)
    ])
    
    X2 = Tensor(randn(Float32, 4, 3, 32, 32))  # 4 imágenes NCHW
    y2 = model2(X2)
    @test size(y2.data) == (10, 4)
    
    println("✓ Integración con Flatten funciona")
end