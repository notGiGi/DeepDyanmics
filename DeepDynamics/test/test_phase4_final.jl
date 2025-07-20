# test_phase4_final.jl
# Test final para verificar que BatchNorm funciona correctamente después del fix

using Test
using DeepDynamics
using Statistics
using DeepDynamics.TensorEngine
using LinearAlgebra
println("="^70)
println("TEST FINAL FASE 4 - VERIFICACIÓN DE CONVERGENCIA CON FIX")
println("="^70)

@testset "Fase 4 FINAL - BatchNorm Convergencia" begin
    
    # Test 1: Verificar que collect_parameters incluye BatchNorm
    @testset "collect_parameters corregido" begin
        println("\n1️⃣ Verificando collect_parameters")
        
        model = Sequential([
            Dense(10, 20),
            BatchNorm(20),
            Activation(relu),
            Dense(20, 10),
            BatchNorm(10),
            Activation(relu),
            Dense(10, 2)
        ])
        
        params = collect_parameters(model)
        
        # Debe haber: 3 Dense (6 params) + 2 BatchNorm (4 params) = 10 total
        @test length(params) == 10
        
        # Verificar que gamma y beta están incluidos
        bn1 = model.layers[2]
        bn2 = model.layers[5]
        
        @test bn1.gamma in params
        @test bn1.beta in params
        @test bn2.gamma in params
        @test bn2.beta in params
        
        println("  ✅ collect_parameters incluye parámetros de BatchNorm")
    end
    
    # Test 2: Convergencia en problema simple
    @testset "Convergencia simple" begin
        println("\n2️⃣ Test de convergencia en problema simple")
        
        # Datos linealmente separables
        X = Tensor(Float32[
            -2 -1.5 -1 -0.5 0.5 1 1.5 2;
            -1 -0.5  0  0.5 0.5 0 -0.5 -1
        ])
        
        y = Tensor(Float32[
            1 1 1 1 0 0 0 0;
            0 0 0 0 1 1 1 1
        ])
        
        model = Sequential([
            Dense(2, 8),
            BatchNorm(8),
            Activation(relu),
            Dense(8, 4),
            BatchNorm(4),
            Activation(relu),
            Dense(4, 2),
            Activation(softmax)
        ])
        
        opt = Adam(learning_rate=0.01f0)
        params = collect_parameters(model)
        
        losses = Float32[]
        
        for epoch in 1:200
            for p in params
                zero_grad!(p)
            end
            
            pred = model(X)
            loss = categorical_crossentropy(pred, y)
            push!(losses, loss.data[1])
            
            backward(loss, [1.0f0])
            optim_step!(opt, params)
        end
        
        initial_loss = losses[1]
        final_loss = losses[end]
        reduction = (1 - final_loss/initial_loss) * 100
        
        println("  Loss inicial: $(round(initial_loss, digits=4))")
        println("  Loss final: $(round(final_loss, digits=4))")
        println("  Reducción: $(round(reduction, digits=1))%")
        
        @test final_loss < initial_loss
        @test reduction > 20  # Al menos 20% de mejora
        
        # Verificar accuracy
        pred_final = model(X)
        pred_classes = argmax(pred_final.data, dims=1)
        true_classes = argmax(y.data, dims=1)
        accuracy = sum(pred_classes .== true_classes) / length(true_classes)
        
        println("  Accuracy final: $(round(accuracy*100, digits=1))%")
        @test accuracy > 0.7  # Al menos 70% accuracy
    end
    
    # Test 3: Comparación con y sin BatchNorm
    @testset "Con vs Sin BatchNorm" begin
        println("\n3️⃣ Comparación con y sin BatchNorm")
        
        # Generar datos más complejos (espiral)
        n_points = 50
        θ = range(0, 4π, length=n_points)
        r = θ / (4π)
        
        X1 = hcat(r .* cos.(θ), r .* sin.(θ))'
        X2 = hcat(-r .* cos.(θ), -r .* sin.(θ))'
        X = Tensor(Float32.(hcat(X1, X2)))
        
        y1 = zeros(Float32, 2, n_points)
        y1[1, :] .= 1
        y2 = zeros(Float32, 2, n_points)
        y2[2, :] .= 1
        y = Tensor(hcat(y1, y2))
        
        # Modelo sin BatchNorm
        model_no_bn = Sequential([
            Dense(2, 32),
            Activation(relu),
            Dense(32, 16),
            Activation(relu),
            Dense(16, 2),
            Activation(softmax)
        ])
        
        # Modelo con BatchNorm
        model_with_bn = Sequential([
            Dense(2, 32),
            BatchNorm(32),
            Activation(relu),
            Dense(32, 16),
            BatchNorm(16),
            Activation(relu),
            Dense(16, 2),
            Activation(softmax)
        ])
        
        # Entrenar ambos
        epochs = 100
        lr = 0.01f0
        
        # Sin BatchNorm
        opt1 = Adam(learning_rate=lr)
        params1 = collect_parameters(model_no_bn)
        final_loss_no_bn = 0.0f0
        
        for e in 1:epochs
            for p in params1
                zero_grad!(p)
            end
            pred = model_no_bn(X)
            loss = categorical_crossentropy(pred, y)
            if e == epochs
                final_loss_no_bn = loss.data[1]
            end
            backward(loss, [1.0f0])
            optim_step!(opt1, params1)
        end
        
        # Con BatchNorm
        opt2 = Adam(learning_rate=lr)
        params2 = collect_parameters(model_with_bn)
        final_loss_with_bn = 0.0f0
        
        for e in 1:epochs
            for p in params2
                zero_grad!(p)
            end
            pred = model_with_bn(X)
            loss = categorical_crossentropy(pred, y)
            if e == epochs
                final_loss_with_bn = loss.data[1]
            end
            backward(loss, [1.0f0])
            optim_step!(opt2, params2)
        end
        
        println("  Loss final sin BatchNorm: $(round(final_loss_no_bn, digits=4))")
        println("  Loss final con BatchNorm: $(round(final_loss_with_bn, digits=4))")
        
        # BatchNorm debería dar similar o mejor resultado
        @test final_loss_with_bn < final_loss_no_bn * 1.2  # Permitimos hasta 20% peor
    end
    
    # Test 4: Estabilidad numérica
    @testset "Estabilidad numérica" begin
        println("\n4️⃣ Test de estabilidad numérica")
        
        # Datos con diferentes escalas
        X_large = Tensor(randn(Float32, 10, 32) .* 100.0f0)
        y_dummy = Tensor(Float32.(rand(0:1, 2, 32)))
        
        model = Sequential([
            Dense(10, 20),
            BatchNorm(20),  # Debe normalizar la gran escala
            Activation(relu),
            Dense(20, 2),
            Activation(softmax)
        ])
        
        # Forward pass
        output = model(X_large)
        
        # No debe haber NaN o Inf
        @test !any(isnan.(output.data))
        @test !any(isinf.(output.data))
        
        # Backward pass
        loss = categorical_crossentropy(output, y_dummy)
        params = collect_parameters(model)
        
        for p in params
            zero_grad!(p)
        end
        
        backward(loss, [1.0f0])
        
        # Gradientes no deben explotar
        for p in params
            if p.grad !== nothing
                @test !any(isnan.(p.grad.data))
                @test !any(isinf.(p.grad.data))
                @test maximum(abs.(p.grad.data)) < 1000  # Gradientes acotados
            end
        end
        
        println("  ✅ BatchNorm mantiene estabilidad numérica")
    end
end

println("\n" * "="^70)
println("✨ FASE 4 COMPLETADA EXITOSAMENTE ✨")
println("BatchNorm ahora funciona correctamente y los modelos convergen")
println("="^70)