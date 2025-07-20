# Tests Completos para BatchNorm - Fase 4
using Test
using DeepDynamics
using Statistics
using CUDA
using DeepDynamics.TensorEngine
println("="^70)
println("TESTS COMPLETOS - FASE 4: BATCHNORM MEJORADO")
println("="^70)

@testset "BatchNorm Fase 4 - Completo" begin
    
    # Test 1: Verificación de dimensiones y formato NCHW
    @testset "1. Dimensiones NCHW" begin
        println("\n🧪 Test 1: Verificación de dimensiones NCHW")
        
        # Datos NCHW
        N, C, H, W = 4, 3, 8, 8
        x = Tensor(randn(Float32, N, C, H, W))
        bn = BatchNorm(C)
        
        y = bn(x)
        @test size(y.data) == (N, C, H, W)
        
        # Verificar que las estadísticas tienen el tamaño correcto
        @test length(bn.running_mean) == C
        @test length(bn.running_var) == C
        
        println("  ✅ Dimensiones preservadas correctamente")
    end
    
    # Test 2: Cálculo correcto de estadísticas
    @testset "2. Estadísticas correctas" begin
        println("\n🧪 Test 2: Cálculo de estadísticas por canal")
        
        # Crear datos con valores conocidos por canal
        N, C, H, W = 2, 3, 4, 4
        x_data = zeros(Float32, N, C, H, W)
        
        # Canal 1: todos 1.0
        x_data[:, 1, :, :] .= 1.0f0
        # Canal 2: todos 2.0
        x_data[:, 2, :, :] .= 2.0f0
        # Canal 3: todos 3.0
        x_data[:, 3, :, :] .= 3.0f0
        
        x = Tensor(x_data)
        bn = BatchNorm(C, momentum=0.1f0, training=true)
        
        # Forward
        y = bn(x)
        
        # Verificar running stats
        # Con momentum=0.1 (convención PyTorch): new = (1-0.1)*0 + 0.1*[1,2,3] = [0.1, 0.2, 0.3]
        expected_mean = [0.1f0, 0.2f0, 0.3f0]
        @test all(abs.(bn.running_mean .- expected_mean) .< 1e-5)
        
        # La varianza debe ser 0 para datos constantes
        # running_var = 0.9*1 + 0.1*0 = 0.9
        expected_var = [0.9f0, 0.9f0, 0.9f0]
        @test all(abs.(bn.running_var .- expected_var) .< 1e-5)
        
        println("  ✅ Estadísticas calculadas correctamente")
    end
    
    # Test 3: Normalización correcta
    @testset "3. Normalización" begin
        println("\n🧪 Test 3: Verificación de normalización")
        
        # Datos aleatorios con media y varianza conocidas
        N, C, H, W = 8, 2, 16, 16
        x_data = randn(Float32, N, C, H, W)
        
        # Escalar cada canal diferente
        x_data[:, 1, :, :] = x_data[:, 1, :, :] .* 2.0f0 .+ 3.0f0  # mean≈3, std≈2
        x_data[:, 2, :, :] = x_data[:, 2, :, :] .* 0.5f0 .- 1.0f0  # mean≈-1, std≈0.5
        
        x = Tensor(x_data)
        bn = BatchNorm(C, training=true)
        
        y = bn(x)
        y_data = y.data
        
        # Verificar normalización por canal
        for c in 1:C
            channel_data = vec(y_data[:, c, :, :])
            chan_mean = mean(channel_data)
            chan_std = std(channel_data)
            
            @test abs(chan_mean) < 0.01  # mean ≈ 0
            @test abs(chan_std - 1.0) < 0.01  # std ≈ 1
            
            println("  Canal $c: mean=$(round(chan_mean, digits=4)), std=$(round(chan_std, digits=4))")
        end
        
        println("  ✅ Normalización correcta")
    end
    
    # Test 4: Modo train vs eval
    @testset "4. Train vs Eval" begin
        println("\n🧪 Test 4: Comportamiento train vs eval")
        
        x = Tensor(randn(Float32, 4, 2, 8, 8))
        bn = BatchNorm(2, momentum=0.1f0, training=true)
        
        # Entrenar varias iteraciones
        for i in 1:5
            _ = bn(x)
        end
        
        # Guardar estadísticas
        train_mean = copy(bn.running_mean)
        train_var = copy(bn.running_var)
        
        # Cambiar a eval
        set_training!(bn, false)
        
        # Datos muy diferentes
        x_test = Tensor(randn(Float32, 4, 2, 8, 8) .* 10.0f0)
        _ = bn(x_test)
        
        # Las estadísticas NO deben cambiar en eval
        @test bn.running_mean == train_mean
        @test bn.running_var == train_var
        
        println("  ✅ Running stats no cambian en eval")
    end
    
    # Test 5: GPU compatibility
    @testset "5. GPU Support" begin
        println("\n🧪 Test 5: Compatibilidad GPU")
        
        if CUDA.functional()
            # Preparar datos y modelo
            x_cpu = Tensor(randn(Float32, 2, 3, 4, 4))
            bn = BatchNorm(3, training=true)
            
            # Forward en CPU
            y_cpu = bn(x_cpu)
            
            # Mover a GPU
            x_gpu = to_gpu(x_cpu)
            
            # Forward en GPU
            y_gpu = bn(x_gpu)
            
            # Verificar que el output está en GPU
            @test y_gpu.data isa CUDA.CuArray
            
            # Running stats deben seguir en CPU
            @test !(bn.running_mean isa CUDA.CuArray)
            @test !(bn.running_var isa CUDA.CuArray)
            
            # Los resultados deben ser similares
            @test isapprox(Array(y_gpu.data), y_cpu.data, rtol=1e-5)
            
            println("  ✅ GPU forward funciona correctamente")
            
            # Test backward en GPU
            loss = mse_loss(y_gpu, to_gpu(Tensor(randn(Float32, 2, 3, 4, 4))))
            zero_grad!(bn.gamma)
            zero_grad!(bn.beta)
            backward(loss, [1.0f0])
            
            @test bn.gamma.grad !== nothing
            @test bn.beta.grad !== nothing
            
            println("  ✅ GPU backward funciona correctamente")
        else
            println("  ⚠️  GPU no disponible, test omitido")
        end
    end
    
    # Test 6: Gradientes correctos
    @testset "6. Gradientes" begin
        println("\n🧪 Test 6: Verificación de gradientes")
        
        # Datos pequeños para verificación numérica
        x = Tensor(randn(Float32, 2, 2, 3, 3); requires_grad=true)
        bn = BatchNorm(2, training=true)
        
        # Forward
        y = bn(x)
        
        # Loss simple
        loss = sum(y.data)
        
        # Backward
        zero_grad!(x)
        zero_grad!(bn.gamma)
        zero_grad!(bn.beta)
        
        # Crear loss tensor apropiadamente
        loss_tensor = Tensor([loss]; requires_grad=true)
        loss_tensor.backward_fn = _ -> begin
            TensorEngine.backward(y, ones(size(y.data)))
        end
        
        backward(loss_tensor, [1.0f0])
        
        # Verificar que todos tienen gradientes
        @test x.grad !== nothing
        @test bn.gamma.grad !== nothing
        @test bn.beta.grad !== nothing
        
        # Verificar que no hay NaN
        @test !any(isnan.(x.grad.data))
        @test !any(isnan.(bn.gamma.grad.data))
        @test !any(isnan.(bn.beta.grad.data))
        
        println("  ✅ Gradientes calculados sin NaN")
    end
    
    # Test 7: BatchNorm 2D para Dense layers
    @testset "7. BatchNorm 2D" begin
        println("\n🧪 Test 7: BatchNorm para capas Dense")
        
        # Formato (features, batch)
        F, B = 10, 32
        x = Tensor(randn(Float32, F, B))
        bn = BatchNorm(F, training=true)
        
        y = bn(x)
        @test size(y.data) == (F, B)
        
        # Verificar normalización
        for f in 1:F
            feature_data = y.data[f, :]
            @test abs(mean(feature_data)) < 0.01
            @test abs(std(feature_data) - 1.0) < 0.02  # tolerancia relajada
        end
        
        println("  ✅ BatchNorm 2D funciona correctamente")
    end
    
    # Test 8: Reset de estadísticas
    @testset "8. Reset stats" begin
        println("\n🧪 Test 8: Reset de estadísticas")
        
        bn = BatchNorm(3, training=true)
        
        # Hacer algunos forward passes
        for _ in 1:10
            x = Tensor(randn(Float32, 4, 3, 8, 8))
            _ = bn(x)
        end
        
        @test bn.num_batches_tracked == 10
        @test !all(bn.running_mean .== 0)
        
        # Reset
        reset_running_stats!(bn)
        
        @test all(bn.running_mean .== 0)
        @test all(bn.running_var .== 1)
        @test bn.num_batches_tracked == 0
        
        println("  ✅ Reset funciona correctamente")
    end
end

# Test de integración final

println("TEST DE INTEGRACIÓN FINAL")


@testset "Integración CNN con BatchNorm" begin
    # Crear un modelo CNN simple
    model = Sequential([
        Conv2D(3, 16, (3,3), padding=(1,1)),
        BatchNorm(16),
        Activation(relu),
        MaxPooling((2,2)),
        Conv2D(16, 32, (3,3), padding=(1,1)),
        BatchNorm(32),
        Activation(relu),
        MaxPooling((2,2)),
        Flatten(),
        Dense(32*16*16, 64),
        BatchNorm(64),
        Activation(relu),
        Dense(64, 10),
        Activation(softmax)
    ])
    
    # Datos de entrada
    X = Tensor(randn(Float32, 4, 3, 64, 64))  # 4 imágenes de 64x64x3
    
    # Forward pass
    output = model(X)
    @test size(output.data) == (10, 4)
    
    # Crear target one-hot
    y = zeros(Float32, 10, 4)
    for i in 1:4
        y[rand(1:10), i] = 1.0f0
    end
    y_tensor = Tensor(y)
    
    # Loss
    loss = categorical_crossentropy(output, y_tensor)
    @test !isnan(loss.data[1])
    
    # Backward
    params = collect_parameters(model)
    for p in params
        zero_grad!(p)
    end
    
    backward(loss, [1.0f0])
    
    # Verificar gradientes en BatchNorm layers
    bn_count = 0
    for layer in model.layers
        if layer isa BatchNorm
            bn_count += 1
            @test layer.gamma.grad !== nothing
            @test layer.beta.grad !== nothing
            @test !any(isnan.(layer.gamma.grad.data))
            @test !any(isnan.(layer.beta.grad.data))
        end
    end
    
    @test bn_count == 3  # Deberíamos tener 3 BatchNorm layers
    
    println("✅ Integración completa exitosa")
end

# Resumen final

println("✨ TODOS LOS TESTS DE BATCHNORM FASE 4 COMPLETADOS ✨")
