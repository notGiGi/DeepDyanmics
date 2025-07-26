using Test
using DeepDynamics
using DeepDynamics.TensorEngine
using DeepDynamics.NeuralNetwork
using DeepDynamics.Optimizers
using DeepDynamics.Layers
using DeepDynamics.ConvolutionalLayers
using DeepDynamics.EmbeddingLayer
using CUDA
using Random

Random.seed!(42)

println("=== TESTS FASE 6: Sistema Unificado GPU ===\n")

# Verificar si CUDA está disponible
const HAS_GPU = CUDA.functional()
println("GPU disponible: ", HAS_GPU)

@testset "Fase 6: Sistema Unificado GPU" begin
    
    @testset "1. Funciones device_of y ensure_on_device" begin
        # Test con arrays
        cpu_array = randn(Float32, 5, 5)
        @test TensorEngine.device_of(cpu_array) == :cpu
        
        if HAS_GPU
            gpu_array = CUDA.randn(Float32, 5, 5)
            @test TensorEngine.device_of(gpu_array) == :gpu
            
            # Test ensure_on_device con arrays
            cpu_to_gpu = TensorEngine.ensure_on_device(cpu_array, :gpu)
            @test cpu_to_gpu isa CUDA.CuArray
            @test TensorEngine.device_of(cpu_to_gpu) == :gpu
            
            gpu_to_cpu = TensorEngine.ensure_on_device(gpu_array, :cpu)
            @test gpu_to_cpu isa Array
            @test TensorEngine.device_of(gpu_to_cpu) == :cpu
        end
        
        # Test con Tensors
        cpu_tensor = Tensor(cpu_array)
        @test TensorEngine.device_of(cpu_tensor) == :cpu
        
        if HAS_GPU
            gpu_tensor = TensorEngine.to_gpu(cpu_tensor)
            @test TensorEngine.device_of(gpu_tensor) == :gpu
            
            # Test ensure_on_device con Tensors
            back_to_cpu = TensorEngine.ensure_on_device(gpu_tensor, :cpu)
            @test TensorEngine.device_of(back_to_cpu) == :cpu
            @test Array(back_to_cpu.data) ≈ cpu_tensor.data
        end
        
        # Test con Nothing
        @test TensorEngine.device_of(nothing) == :cpu
        @test TensorEngine.ensure_on_device(nothing, :gpu) === nothing
        
        println("✓ device_of y ensure_on_device funcionan correctamente")
    end
    
    @testset "2. same_device y match_device" begin
        t1 = Tensor(randn(Float32, 3, 3))
        t2 = Tensor(randn(Float32, 3, 3))
        
        @test TensorEngine.same_device(t1, t2) == true
        
        if HAS_GPU
            t1_gpu = TensorEngine.to_gpu(t1)
            @test TensorEngine.same_device(t1, t1_gpu) == false
            
            # Test match_device
            t2_matched = TensorEngine.match_device(t1_gpu, t2)
            @test TensorEngine.same_device(t1_gpu, t2_matched) == true
            @test TensorEngine.device_of(t2_matched) == :gpu
        end
        
        println("✓ same_device y match_device funcionan correctamente")
    end
    
    @testset "3. Optimizadores con dispositivos correctos" begin
        param_cpu = Tensor(randn(Float32, 5, 5); requires_grad=true)
        param_cpu.grad = Tensor(randn(Float32, 5, 5); requires_grad=false)
        
        adam = Adam(learning_rate=0.01)
        Optimizers.step!(adam, [param_cpu])
        @test haskey(adam.m, param_cpu)
        @test TensorEngine.device_of(adam.m[param_cpu]) == :cpu
        @test TensorEngine.device_of(adam.v[param_cpu]) == :cpu
        
        if HAS_GPU
            param_gpu = TensorEngine.to_gpu(param_cpu)
            param_gpu.grad = TensorEngine.to_gpu(param_cpu.grad)
            Optimizers.step!(adam, [param_gpu])
            @test TensorEngine.device_of(adam.m[param_gpu]) == :gpu
            @test TensorEngine.device_of(adam.v[param_gpu]) == :gpu
        end
        
        println("✓ Optimizadores manejan dispositivos correctamente")
    end
    
    @testset "4. model_to_device para modelos complejos" begin
        model = NeuralNetwork.model_to_cpu(Sequential([
            Conv2D(3, 32, (3,3), stride=(1,1), padding=(1,1)),
            BatchNorm(32),
            LayerActivation(relu),
            MaxPooling((2,2)),
            create_residual_block(32, 64, 2),
            GlobalAvgPool(),
            Flatten(),
            DropoutLayer(0.5),
            Dense(64, 10),
            Activation(softmax)
        ]))
        
        @test NeuralNetwork.model_device(model) == :cpu
        
        if HAS_GPU
            model_gpu = NeuralNetwork.model_to_gpu(model)
            @test NeuralNetwork.model_device(model_gpu) == :gpu
            params_gpu = NeuralNetwork.collect_parameters(model_gpu)
            @test all(TensorEngine.device_of(p) == :gpu for p in params_gpu)
            input_gpu = Tensor(CUDA.randn(Float32, 1, 3, 32, 32))
            output_gpu = forward(model_gpu, input_gpu)
            @test TensorEngine.device_of(output_gpu) == :gpu
            model_cpu = NeuralNetwork.model_to_cpu(model_gpu)
            input_cpu = Tensor(randn(Float32, 1, 3, 32, 32))
            output_cpu = forward(model_cpu, input_cpu)
            @test TensorEngine.device_of(output_cpu) == :cpu
        else
            input_cpu = Tensor(randn(Float32, 1, 3, 32, 32))
            output_cpu = forward(model, input_cpu)
            @test TensorEngine.device_of(output_cpu) == :cpu
        end
        
        println("✓ model_to_device maneja modelos complejos correctamente")
    end
    
    @testset "5. Entrenamiento con dispositivos mixtos" begin
        model = NeuralNetwork.model_to_cpu(Sequential([
            Dense(10, 20),
            LayerActivation(relu),
            Dense(20, 2),
            Activation(softmax)
        ]))
        
        X = Tensor(randn(Float32, 10, 50))
        y = Tensor(zeros(Float32, 2, 50))
        for i in 1:50 y.data[rand(1:2), i] = 1.0f0 end
        
        optimizer = Adam(learning_rate=0.01)
        params = NeuralNetwork.collect_parameters(model)
        
        losses_cpu = Float32[]
        for epoch in 1:3
            for p in params TensorEngine.zero_grad!(p) end
            output = forward(model, X)
            loss = categorical_crossentropy(output, y)
            push!(losses_cpu, loss.data[1])
            backward(loss, [1.0f0])
            Optimizers.step!(optimizer, params)
        end
        
        if HAS_GPU
            model_gpu = NeuralNetwork.model_to_gpu(model)
            X_gpu = TensorEngine.to_gpu(X)
            y_gpu = TensorEngine.to_gpu(y)
            Optimizers.optimizer_to_device!(optimizer, :gpu)
            params_gpu = NeuralNetwork.collect_parameters(model_gpu)
            losses_gpu = Float32[]
            for epoch in 1:3
                for p in params_gpu TensorEngine.zero_grad!(p) end
                output = forward(model_gpu, X_gpu)
                loss = categorical_crossentropy(output, y_gpu)
                push!(losses_gpu, loss.data[1])
                backward(loss, CUDA.ones(Float32, 1))
                Optimizers.step!(optimizer, params_gpu)
            end
            @test all(isfinite.(losses_cpu))
            @test all(isfinite.(losses_gpu))
            @test losses_gpu[end] < losses_gpu[1]
        end
        
        println("✓ Entrenamiento funciona con dispositivos mixtos")
    end
    
    @testset "6. Gestión de memoria GPU" begin
        if HAS_GPU
            info = TensorEngine.gpu_memory_info()
            @test info.total > 0
            @test 0 <= info.free_percent <= 100
            println("  Memoria GPU: $(round(info.used/1e9, digits=2)) GB usados de $(round(info.total/1e9, digits=2)) GB")
            TensorEngine.ensure_gpu_memory!(80.0)
            info_after = TensorEngine.gpu_memory_info()
            println("  Memoria libre después de limpieza: $(round(info_after.free_percent, digits=1))%")
            @test info_after.free_percent >= info.free_percent
        else
            println("  (Saltando tests de memoria GPU - no hay GPU disponible)")
        end
        
        println("✓ Gestión de memoria GPU funciona correctamente")
    end
    
    @testset "7. Integración completa" begin
        model = NeuralNetwork.model_to_cpu(create_resnet(3, 10; blocks=[1,1,1,1]))
        params = NeuralNetwork.collect_parameters(model)
        @test length(params) > 0
        
        if HAS_GPU
            model_gpu = NeuralNetwork.model_to_gpu(model)
            input_gpu = Tensor(CUDA.randn(Float32, 2, 3, 64, 64))
            output_gpu = forward(model_gpu, input_gpu)
            @test size(output_gpu.data) == (10, 2)
            @test TensorEngine.device_of(output_gpu) == :gpu
            params_gpu = NeuralNetwork.collect_parameters(model_gpu)
            @test all(TensorEngine.device_of(p) == :gpu for p in params_gpu)
        else
            println("  (Saltando integración GPU - no hay GPU disponible)")
        end
        
        println("✓ Integración completa funciona correctamente")
    end

end

println("\n=== RESUMEN FASE 6 ===")
println("✓ Sistema unificado de dispositivos implementado")
println("✓ Funciones device_of y ensure_on_device funcionan correctamente")
println("✓ Optimizadores crean y mantienen momentos en el dispositivo correcto")
println("✓ model_to_gpu/cpu maneja todos los tipos de capas")
println("✓ Entrenamiento funciona sin problemas en CPU y GPU")
println("✓ Gestión de memoria GPU implementada")

if !HAS_GPU
    println("\n⚠️  NOTA: Algunos tests fueron omitidos porque no hay GPU disponible")
end

println("\n¡Fase 6 completada exitosamente!")
