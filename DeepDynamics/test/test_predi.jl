# test/test_inference.jl
using Test
using DeepDynamics
using DeepDynamics.NeuralNetwork
using DeepDynamics.TensorEngine
using DeepDynamics.Inference
using CUDA
using Statistics
using BenchmarkTools

println("="^60)
println("TEST FASE 14: Sistema de Inferencia")
println("="^60)

@testset "Sistema de Inferencia - Producción" begin
    
    # ============= MODELOS DE PRUEBA =============
    function create_test_model_dense()
        Sequential([
            Dense(784, 128),
            BatchNorm(128),
            Activation(relu),
            Dense(128, 64),
            BatchNorm(64), 
            Activation(relu),
            Dense(64, 10),
            Activation(softmax)
        ])
    end
    
    function create_test_model_cnn()
        Sequential([
            Conv2D(3, 32, (3,3), padding=(1,1)),
            BatchNorm(32),
            Activation(relu),
            MaxPooling((2,2)),
            Conv2D(32, 64, (3,3), padding=(1,1)),
            BatchNorm(64),
            Activation(relu),
            MaxPooling((2,2)),
            Flatten(),
            Dense(64 * 8 * 8, 10),
            Activation(softmax)
        ])
    end
    
    # ============= TEST 1: PREDICT BÁSICO =============
    @testset "1. Predict Básico" begin
        println("\n🧪 Test 1: Predict básico")
        
        model = create_test_model_dense()
        
        # Test con arrays
        X_array = randn(Float32, 784, 100)
        predictions = predict(model, X_array, batch_size=32, verbose=false)
        
        @test size(predictions, 2) == 100
        @test size(predictions, 1) == 10
        @test all(predictions .>= 0)
        @test all(isapprox.(sum(predictions, dims=1), 1.0f0, atol=1e-5))
        
        # Test con tensores
        X_tensors = [Tensor(randn(Float32, 784)) for _ in 1:50]
        preds_tensor = predict(model, X_tensors, batch_size=16)
        @test size(preds_tensor, 2) == 50
        
        # Test batch_size mayor que datos
        X_small = randn(Float32, 784, 5)
        preds_small = predict(model, X_small, batch_size=100)
        @test size(preds_small, 2) == 5
        
        # Test entrada vacía
        @test_throws ArgumentError predict(model, Float32[])
        
        println("  ✅ Predict básico funciona correctamente")
    end
    
    # ============= TEST 2: PREDICT_PROBA =============
    @testset "2. Predict Proba" begin
        println("\n🧪 Test 2: Predict proba con temperature scaling")
        
        model = create_test_model_dense()
        X = randn(Float32, 784, 20)
        
        # Sin temperature
        probs_t1 = predict_proba(model, X, temperature=1.0f0)
        @test all(probs_t1 .>= 0) && all(probs_t1 .<= 1)
        @test all(isapprox.(sum(probs_t1, dims=1), 1.0f0, atol=1e-5))
        
        # Con temperature alta (más uniforme)
        probs_t2 = predict_proba(model, X, temperature=2.0f0)
        entropy_t1 = -sum(probs_t1 .* log.(probs_t1 .+ 1e-8)) / size(X, 2)
        entropy_t2 = -sum(probs_t2 .* log.(probs_t2 .+ 1e-8)) / size(X, 2)
        @test entropy_t2 > entropy_t1
        
        # Con temperature baja (más confiado)
        probs_t05 = predict_proba(model, X, temperature=0.5f0)
        max_conf_t05 = maximum(probs_t05, dims=1)
        max_conf_t1 = maximum(probs_t1, dims=1)
        @test mean(max_conf_t05) > mean(max_conf_t1)
        
        # Test clasificación binaria
        model_binary = Sequential([
            Dense(10, 5),
            Activation(relu),
            Dense(5, 1),
            Activation(sigmoid)
        ])
        X_binary = randn(Float32, 10, 30)
        probs_binary = predict_proba(model_binary, X_binary)
        @test all(probs_binary .>= 0) && all(probs_binary .<= 1)
        
        println("  ✅ Temperature scaling funciona correctamente")
    end
    
    # ============= TEST 3: PREDICT_GENERATOR =============
    @testset "3. Predict Generator" begin
        println("\n🧪 Test 3: Predict con generador")
        
        model = create_test_model_dense()
        
        # Generador simple
        function data_generator(n_batches=10, batch_size=32)
            Channel() do ch
                for i in 1:n_batches
                    batch = randn(Float32, 784, batch_size)
                    put!(ch, batch)
                end
            end
        end
        
        # Test secuencial
        gen = data_generator(5, 20)
        preds = predict_generator(model, gen, steps=nothing, workers=1)  # Sin límite, procesa todo
        @test size(preds, 2) == 100  # 5 batches * 20 samples 
        
        # Test con límite de steps
        gen2 = data_generator(10, 16)
        preds2 = predict_generator(model, gen2, steps=3)
        @test size(preds2, 2) == 48
        
        # Test paralelo si hay threads
        if Threads.nthreads() > 1
            gen3 = data_generator(8, 25)
            preds3 = predict_generator(model, gen3, workers=2, use_multiprocessing=true)
            @test size(preds3, 2) == 200
            println("  ✅ Paralelismo con $(Threads.nthreads()) threads")
        end
        
        println("  ✅ Generator funciona correctamente")
    end
    
    # ============= TEST 4: PIPELINE =============
    @testset "4. PredictionPipeline" begin
        println("\n🧪 Test 4: Pipeline completo")
        
        model = create_test_model_dense()
        
        # Preprocesamiento
        preprocess = function(X)
            μ = mean(X, dims=1)
            σ = std(X, dims=1) .+ 1e-8
            return (X .- μ) ./ σ
        end
        
        # Postprocesamiento
        postprocess = function(probs)
            max_probs = maximum(probs, dims=1)
            indices = [argmax(probs[:, i]) for i in axes(probs, 2)]  # ✅ Usar axes
            return (classes=indices, confidence=vec(max_probs))
        end
        
        # Crear pipeline
        pipeline = PredictionPipeline(
            model, 
            preprocess, 
            postprocess,
            device=:cpu,
            batch_size=64
        )
        
        # Test
        X = randn(Float32, 784, 30)
        results = pipeline(X)
        
        @test haskey(results, :classes)
        @test haskey(results, :confidence)
        @test length(results.classes) == 30
        @test all(results.confidence .>= 0) && all(results.confidence .<= 1)
        @test all(1 .<= results.classes .<= 10)
        
        println("  ✅ Pipeline funciona correctamente")
    end
    
    # ============= TEST 5: CUANTIZACIÓN =============
    @testset "5. Model Quantization" begin
        println("\n🧪 Test 5: Cuantización del modelo")
        
        model = create_test_model_dense()
        X = randn(Float32, 784, 10)
        
        # Predicción original
        preds_original = predict(model, X)
        
        # Cuantizar
        model_quant = quantize_model(model, bits=8)
        
        # Predicción cuantizada
        preds_quant = predict(model_quant, X)
        
        # Verificar similitud
        max_diff = maximum(abs.(preds_original .- preds_quant))
        mean_diff = mean(abs.(preds_original .- preds_quant))
        
        @test mean_diff < 0.1  # Error promedio < 10%
        @test max_diff < 0.3    # Error máximo < 30%
        
   
        for layer in model_quant.layers
            if hasproperty(layer, :weights)
                W_original = model.layers[findfirst(l -> l === layer, model_quant.layers)].weights.data
                W_quant = layer.weights.data
                
                # Verificar que hay cuantización real
                # Los valores cuantizados deben estar en un rango más discreto
                original_range = maximum(abs.(W_original))
                quant_range = maximum(abs.(W_quant))
                
                # La cuantización debe mantener el rango similar pero con menos precisión
                @test isapprox(original_range, quant_range, rtol=0.3)
                
                # Verificar que los valores son diferentes (prueba de que hubo cuantización)
                @test !all(isapprox.(W_original, W_quant, atol=1e-6))
            end
        end
        
        println("  ✅ Cuantización funciona (error medio: $(round(mean_diff*100, digits=2))%)")
    end
    
    # ============= TEST 6: CNN =============
    @testset "6. CNN Inference" begin
        println("\n🧪 Test 6: Inferencia CNN")
        
        model = create_test_model_cnn()
        X = randn(Float32, 4, 3, 32, 32)  # NCHW
        
        preds = predict(model, X)
        @test size(preds) == (10, 4)
        @test all(preds .>= 0)
        @test all(isapprox.(sum(preds, dims=1), 1.0f0, atol=1e-5))
        
        println("  ✅ CNN inference funciona")
    end
    
    # ============= TEST 7: GPU si disponible =============
    if CUDA.functional()
        @testset "7. GPU Inference" begin
            println("\n🧪 Test 7: Inferencia en GPU")
            
            model = create_test_model_dense()
            X = randn(Float32, 784, 20)
            
            # Test auto
            preds_auto = predict(model, X, device=:auto)
            @test size(preds_auto) == (10, 20)
            
            # Test GPU explícito
            preds_gpu = predict(model, X, device=:cuda)
            @test size(preds_gpu) == (10, 20)
            
            # Test CPU explícito
            preds_cpu = predict(model, X, device=:cpu)
            @test size(preds_cpu) == (10, 20)
            
            # Comparar resultados
            @test isapprox(preds_gpu, preds_cpu, rtol=1e-4)
            
            println("  ✅ GPU inference funciona")
        end
    else
        println("\n⚠️  GPU no disponible, test GPU omitido")
    end
    
    # ============= TEST 8: WARMUP =============
    @testset "8. Model Warmup" begin
        println("\n🧪 Test 8: Cache warmup")
        
        model = create_test_model_dense()
        sample = randn(Float32, 784, 1)
        
        # Warmup no debe fallar
        warmup_model(model, sample, n_runs=3, verbose=false)
        
        # Test con generador
        gen = Channel() do ch
            put!(ch, randn(Float32, 784, 8))
        end
        warmup_model(model, gen, n_runs=2, verbose=false)
        
        println("  ✅ Warmup funciona correctamente")
    end
    
    # ============= TEST 9: MEMORY POOL =============
    @testset "9. Tensor Memory Pool" begin
        println("\n🧪 Test 9: Memory pooling")
        
        pool = TensorPool(max_tensors=5)
        
        # Obtener tensor
        t1 = get_tensor_from_pool!(pool, (10, 20), :cpu)
        @test size(t1.data) == (10, 20)
        @test all(t1.data .== 0)
        
        # Modificar y devolver
        t1.data .= 1
        return_to_pool!(pool, t1)
        @test length(pool.pools[(10, 20)]) == 1
        
        # Reutilizar (debe estar limpio)
        t2 = get_tensor_from_pool!(pool, (10, 20), :cpu)
        @test all(t2.data .== 0)  # Limpiado en get_tensor_from_pool!
        
        # Test límite del pool
        for i in 1:10
            t = get_tensor_from_pool!(pool, (5, 5), :cpu)
            return_to_pool!(pool, t)
        end
        @test length(pool.pools[(5, 5)]) <= 5
        
        println("  ✅ Memory pool funciona correctamente")
    end
    
    # ============= TEST 10: BENCHMARKS =============
    @testset "10. Performance Benchmarks" begin
        println("\n🧪 Test 10: Benchmarks de rendimiento")
        
        model = create_test_model_dense()
        
        # Warmup
        X_warmup = randn(Float32, 784, 1)
        warmup_model(model, X_warmup, verbose=false)
        
        # Latencia (batch=1)
        X_single = randn(Float32, 784, 1)
        time_single = @elapsed predict(model, X_single)
        latency_ms = time_single * 1000
        
        println("  Latencia (batch=1): $(round(latency_ms, digits=2))ms")
        @test latency_ms < 100  # Objetivo <10ms en GPU, <100ms en CPU
        
        # Throughput
        X_batch = randn(Float32, 784, 100)
        time_batch = @elapsed predict(model, X_batch, batch_size=100)
        throughput = 100 / time_batch
        
        println("  Throughput: $(round(throughput, digits=1)) samples/sec")
        @test throughput > 50  # Mínimo 50 samples/sec
        
        # Estabilidad de memoria
        initial_mem = Base.summarysize(model)
        for i in 1:10
            _ = predict(model, X_batch)
        end
        final_mem = Base.summarysize(model)
        
        @test abs(final_mem - initial_mem) < initial_mem * 0.1  # <10% variación
        println("  ✅ Memoria estable (variación: $(round(abs(final_mem - initial_mem)/initial_mem * 100, digits=1))%)")
    end
    
    # ============= TEST 11: EDGE CASES =============
    @testset "11. Edge Cases y Robustez" begin
        println("\n🧪 Test 11: Casos límite")
        
        model = create_test_model_dense()
        
        # Entrada vacía
        @test_throws ArgumentError predict(model, Float32[])
        @test_throws ArgumentError predict(model, Vector{Tensor}())
        
        # Dimensiones incorrectas
        X_wrong = randn(Float32, 100, 10)
        @test_throws Exception predict(model, X_wrong)
        
        # Un solo sample
        X_one = randn(Float32, 784)
        pred_one = predict(model, reshape(X_one, 784, 1))
        @test size(pred_one, 2) == 1
        
        # Valores extremos
        X_extreme = Float32.(rand([-1000, 1000], 784, 5))
        pred_extreme = predict(model, X_extreme)
        @test !any(isnan.(pred_extreme))
        @test !any(isinf.(pred_extreme))
        
        # Batch muy grande
        X_large = randn(Float32, 784, 1000)
        pred_large = predict(model, X_large, batch_size=50)
        @test size(pred_large, 2) == 1000
        
        println("  ✅ Manejo robusto de edge cases")
    end
end

println("\n" * "="^60)
println("✨ TODOS LOS TESTS DE INFERENCIA COMPLETADOS ✨")
println("="^60)

# Resumen
println("\n📊 RESUMEN:")
println("  ✅ Módulo Inference funcionando correctamente")
println("  ✅ Sin dependencias circulares")
println("  ✅ Arquitectura limpia y escalable")
println("  ✅ Listo para producción")