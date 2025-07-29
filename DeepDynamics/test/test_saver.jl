# test/test_saver.jl - VERSIÓN CORREGIDA
using Test
using DeepDynamics
using CUDA

@testset "ModelSaver Tests" begin
    # Crear modelo de prueba
    model = Sequential([
        Dense(10, 20),
        BatchNorm(20),
        Activation(relu),
        DropoutLayer(0.5),
        Dense(20, 10),
        Activation(softmax)
    ])
    
    # Datos de prueba
    x = Tensor(randn(Float32, 10, 5))
    
    # Test 1: Guardar y cargar modelo
    @testset "Save/Load Model" begin
        filepath = "test_model.jld"
        
        # Poner modelo en modo eval para resultados consistentes
        set_training_mode!(model, false)
        y = model(x)
        
        # Guardar
        save_model(filepath, model; metadata=Dict("test" => "value"))
        @test isfile(filepath)
        
        # Cargar
        loaded_model = load_model(filepath)
        set_training_mode!(loaded_model, false)  # Asegurar modo eval
        
        # Verificar que produce los mismos resultados
        y_loaded = loaded_model(x)
        @test isapprox(y.data, y_loaded.data, rtol=1e-5)
        
        # Limpiar
        rm(filepath)
    end
    
    # Test 2: Checkpoint completo
    @testset "Save/Load Checkpoint" begin
        # Preparar optimizador y métricas
        opt = Adam(learning_rate=0.001)
        params = collect_parameters(model)
        
        # Simular un paso de entrenamiento para que Adam tenga estado
        for p in params
            p.grad = Tensor(randn(Float32, size(p.data)...))
        end
        optim_step!(opt, params)
        
        epoch = 10
        metrics = Dict(
            "train_loss" => [0.5, 0.4, 0.3],
            "val_loss" => [0.6, 0.5, 0.4],
            "train_acc" => [0.8, 0.85, 0.9]
        )
        
        filepath = "test_checkpoint.jld"
        
        # Guardar checkpoint
        save_checkpoint(filepath, model, opt, epoch, metrics)
        @test isfile(filepath)
        
        # Cargar checkpoint
        loaded_model, loaded_opt, loaded_epoch, loaded_metrics = load_checkpoint(filepath)
        
        @test loaded_epoch == epoch
        @test loaded_metrics == metrics
        @test loaded_opt.learning_rate == opt.learning_rate
        
        # Verificar modelo en modo eval
        set_training_mode!(model, false)
        set_training_mode!(loaded_model, false)
        y = model(x)
        y_checkpoint = loaded_model(x)
        @test isapprox(y.data, y_checkpoint.data, rtol=1e-5)
        
        # Limpiar
        rm(filepath)
    end
    
    # Test 3: GPU compatibility
    if CUDA.functional()
        @testset "GPU Save/Load" begin
            # Modelo en GPU
            model_gpu = model_to_gpu(model)
            x_gpu = to_gpu(x)
            
            # Modo eval para consistencia
            set_training_mode!(model_gpu, false)
            y_gpu = model_gpu(x_gpu)
            
            filepath = "test_gpu_model.jld"
            
            # Guardar desde GPU
            save_model(filepath, model_gpu)
            
            # Cargar a CPU
            loaded_cpu = load_model(filepath; device="cpu")
            set_training_mode!(loaded_cpu, false)
            @test !(collect_parameters(loaded_cpu)[1].data isa CUDA.CuArray)
            
            # Cargar a GPU
            loaded_gpu = load_model(filepath; device="cuda")
            set_training_mode!(loaded_gpu, false)
            @test collect_parameters(loaded_gpu)[1].data isa CUDA.CuArray
            
            # Verificar resultados
            y_loaded = loaded_gpu(x_gpu)
            @test isapprox(Array(y_gpu.data), Array(y_loaded.data), rtol=1e-5)
            
            rm(filepath)
        end
    end
    
    # Test 4: Modelo CNN
    @testset "CNN Model Save/Load" begin
        cnn = Sequential([
            Conv2D(3, 16, (3,3)),
            BatchNorm(16),
            Activation(relu),
            MaxPooling((2,2)),
            Flatten(),
            Dense(16*15*15, 10),
            Activation(softmax)
        ])
        
        x_cnn = Tensor(randn(Float32, 1, 3, 32, 32))
        
        # Modo eval
        set_training_mode!(cnn, false)
        y_cnn = cnn(x_cnn)
        
        filepath = "test_cnn.jld"
        save_model(filepath, cnn)
        
        loaded_cnn = load_model(filepath)
        set_training_mode!(loaded_cnn, false)
        y_loaded = loaded_cnn(x_cnn)
        
        @test isapprox(y_cnn.data, y_loaded.data, rtol=1e-5)
        
        rm(filepath)
    end
    
    # Test 5: Error handling
    @testset "Error Handling" begin
        # Archivo inexistente
        @test_throws ErrorException load_model("nonexistent.jld")
        
        # Archivo corrupto
        open("corrupt.jld", "w") do io
            write(io, "corrupted data")
        end
        @test_throws ErrorException load_model("corrupt.jld")
        rm("corrupt.jld")
    end
    
    # Test 6: Preservación de estado BatchNorm
    @testset "BatchNorm State Preservation" begin
        # Crear modelo con BatchNorm
        bn_model = Sequential([
            Dense(5, 10),
            BatchNorm(10),
            Activation(relu)
        ])
        
        # Entrenar un poco para cambiar running stats
        set_training_mode!(bn_model, true)
        for i in 1:5
            _ = bn_model(Tensor(randn(Float32, 5, 8)))
        end
        
        # Guardar running stats originales
        bn_layer = bn_model.layers[2]
        original_mean = copy(bn_layer.running_mean)
        original_var = copy(bn_layer.running_var)
        original_batches = bn_layer.num_batches_tracked
        
        # Guardar y cargar
        filepath = "test_bn_state.jld"
        save_model(filepath, bn_model)
        loaded_bn_model = load_model(filepath)
        
        # Verificar que se preservaron los stats
        loaded_bn = loaded_bn_model.layers[2]
        @test isapprox(loaded_bn.running_mean, original_mean, rtol=1e-5)
        @test isapprox(loaded_bn.running_var, original_var, rtol=1e-5)
        @test loaded_bn.num_batches_tracked == original_batches
        
        rm(filepath)
    end
end

println("✅ Todos los tests de ModelSaver pasaron!")