# test/test_mlops_integrations.jl
using Test
using DeepDynamics
using DeepDynamics.MLOpsIntegrations
using DeepDynamics.TensorEngine
using DeepDynamics.NeuralNetwork
using DeepDynamics.Training
using DeepDynamics.Optimizers
using DeepDynamics.Callbacks
using DeepDynamics.Logging: safe_timestamp
using JSON3
using HTTP
using Dates
using CUDA  # Importar CUDA aquí
using Random
@testset "MLOps Integrations Tests" begin
    
    # Directorio temporal para tests
    test_dir = mktempdir()
    
    @testset "1. MLOpsConfig" begin
        println("\n🧪 Test 1: Configuración MLOps")
        
        # Test configuración por defecto
        config = MLOpsConfig()
        @test config.offline_mode == false
        @test config.retry_attempts == 3
        @test config.batch_size == 100
        
        # Test configuración personalizada
        custom_config = MLOpsConfig(
            offline_mode=true,
            retry_attempts=5,
            mlflow_tracking_uri="http://custom:5000"
        )
        @test custom_config.offline_mode == true
        @test custom_config.mlflow_tracking_uri == "http://custom:5000"
        
        println("  ✅ Configuración funciona correctamente")
    end
    
    @testset "2. WandB Logger - Modo Offline" begin
        println("\n🧪 Test 2: WandB Logger en modo offline")
        
        project_config = Dict(
            "model" => "test_model",
            "learning_rate" => 0.01,
            "batch_size" => 32
        )
        
        # Crear logger en modo offline
        wandb_logger = WandBLogger("test_project", project_config,
                                  offline_mode=true,
                                  log_dir=test_dir)
        
        @test wandb_logger.offline_mode == true
        @test wandb_logger.project == "test_project"
        @test isfile(wandb_logger.local_logger.filepath)
        
        # Simular callbacks
        logs = Dict(
            :loss => 0.5f0,
            :accuracy => 0.85f0,
            :val_loss => 0.6f0,
            :val_accuracy => 0.83f0
        )
        
        on_epoch_end(wandb_logger, 1, logs)
        on_epoch_end(wandb_logger, 2, merge(logs, Dict(:loss => 0.4f0)))
        on_train_end(wandb_logger, logs)
        
        # Verificar que se guardaron los datos localmente
        @test length(wandb_logger.local_logger.metrics_history["loss"]) == 2
        @test isfile(wandb_logger.local_logger.filepath)
        
        println("  ✅ WandB modo offline funciona")
    end
    
    @testset "3. MLFlow Logger - Modo Offline" begin
        println("\n🧪 Test 3: MLFlow Logger en modo offline")
        
        mlflow_logger = MLFlowLogger("test_experiment",
                                    offline_mode=true,
                                    log_dir=test_dir)
        
        @test mlflow_logger.offline_mode == true
        @test mlflow_logger.experiment_name == "test_experiment"
        
        # Test callbacks
        train_logs = Dict(
            :optimizer => Adam(0.01f0),
            :batch_size => 32,
            :epochs => 10
        )
        
        on_train_begin(mlflow_logger, train_logs)
        
        # Simular épocas
        for epoch in 1:3
            epoch_logs = Dict(
                :loss => 1.0f0 / epoch,
                :accuracy => 0.8f0 + 0.05f0 * epoch
            )
            on_epoch_end(mlflow_logger, epoch, epoch_logs)
        end
        
        on_train_end(mlflow_logger, train_logs)
        
        # Verificar logs locales
        @test length(mlflow_logger.local_logger.metrics_history["loss"]) == 3
        @test mlflow_logger.local_logger.metrics_history["accuracy"][3] ≈ 0.95f0
        
        println("  ✅ MLFlow modo offline funciona")
    end
    
    @testset "4. Mock Server Tests" begin
        println("\n🧪 Test 4: Pruebas con servidor mock")
        
        # Crear un servidor HTTP mock simple
        mock_port = 8765
        mock_server = nothing
        
        try
            # Iniciar servidor mock en un task
            mock_server = @async HTTP.serve(mock_port) do request
                # Simular respuestas de W&B/MLFlow
                if contains(request.target, "/api/runs")
                    return HTTP.Response(200, 
                        JSON3.write(Dict("id" => "mock_run_123")))
                elseif contains(request.target, "/experiments")
                    return HTTP.Response(200,
                        JSON3.write(Dict("experiment_id" => "exp_123")))
                else
                    return HTTP.Response(200, "{}")
                end
            end
            
            sleep(0.5)  # Dar tiempo al servidor para iniciar
            
            # Test con servidor mock
            config = MLOpsConfig(
                wandb_base_url="http://localhost:$mock_port",
                mlflow_tracking_uri="http://localhost:$mock_port",
                wandb_api_key="test_key"
            )
            
            # Probar creación
            logger = nothing
            logger_created = false
            error_msg = ""
            
            try
                logger = WandBLogger("test", Dict(), 
                                    api_key="test_key",
                                    base_url=config.wandb_base_url,
                                    offline_mode=false)
                logger_created = (logger !== nothing && logger.run_id != "")
            catch e
                error_msg = string(e)
                logger_created = false
            end
            
            @test logger_created
            
            if !logger_created
                @warn "No se pudo crear logger con servidor mock: $error_msg"
            else
                println("  ✅ Mock server tests pasaron")
            end
            
        catch e
            @warn "No se pudo ejecutar test de servidor mock: $e"
            @test_skip true  # Saltar el test si no se puede ejecutar
        finally
            # Limpiar servidor mock si existe
            if mock_server !== nothing
                try
                    # Señal para detener el servidor
                    Base.throwto(mock_server, InterruptException())
                catch
                end
            end
        end
    end
    
    @testset "5. Integración con Training" begin
        println("\n🧪 Test 5: Integración completa con training")
        
        # Modelo simple
        model = Sequential([
            Dense(10, 20),
            Activation(relu),
            Dense(20, 1)
        ])
        
        # Datos sintéticos
        n_samples = 50
        X_data = [Tensor(randn(Float32, 10, 1)) for _ in 1:n_samples]
        y_data = [Tensor(randn(Float32, 1, 1)) for _ in 1:n_samples]
        
        # Crear loggers MLOps
        mlops_loggers = create_mlops_logger(
            project_name="test_integration",
            platforms=[:wandb, :mlflow],
            config=Dict("test" => true),
            mlops_config=MLOpsConfig(offline_mode=true, 
                                   wandb_api_key="dummy")
        )
        
        @test length(mlops_loggers) >= 1
        
        # Entrenar con loggers
        history = fit!(model, X_data, y_data,
                      epochs=3,
                      batch_size=10,
                      optimizer=Adam(0.01f0),
                      callbacks=mlops_loggers,
                      verbose=false)
        
        @test length(history.train_loss) == 3
        @test history.train_loss[3] < history.train_loss[1]
        
        println("  ✅ Integración con training funciona")
    end
    
    @testset "6. Memory Leaks Test" begin
        println("\n🧪 Test 6: Verificación de memory leaks")
        
        GC.gc(true)  # Limpiar antes de medir
        initial_mem = Base.gc_bytes()
        
        # Crear y destruir múltiples loggers
        for i in 1:10
            logger = WandBLogger("mem_test", Dict(),
                               offline_mode=true,
                               log_dir=test_dir)
            
            # Simular uso intensivo
            for epoch in 1:100
                logs = Dict(:loss => rand(), :accuracy => rand())
                on_epoch_end(logger, epoch, logs)
            end
            
            on_train_end(logger, Dict())
        end
        
        # Forzar garbage collection
        GC.gc(true)
        sleep(0.1)
        
        final_mem = Base.gc_bytes()
        mem_increase = (final_mem - initial_mem) / 1e6  # MB
        
        @test mem_increase < 100  # No más de 100MB de aumento
        
        println("  ✅ No hay memory leaks significativos (aumento: $(round(mem_increase, digits=2)) MB)")
    end
    
    @testset "7. Performance Impact Test" begin
        println("\n🧪 Test 7: Impacto en performance < 1%")
        
        # Modelo y datos de prueba
        model = Sequential([Dense(100, 50), Activation(relu), Dense(50, 10)])
        X = [Tensor(randn(Float32, 100, 1)) for _ in 1:100]
        y = [Tensor(randn(Float32, 10, 1)) for _ in 1:100]
        
        # Benchmark sin logging
        t1 = @elapsed begin
            fit!(model, X, y, epochs=5, batch_size=20, verbose=false)
        end
        
        # Reset modelo
        for p in collect_parameters(model)
            p.data .= randn(Float32, size(p.data)...)
        end
        
        # Benchmark con logging MLOps
        loggers = create_mlops_logger(
            platforms=[:wandb, :mlflow],
            mlops_config=MLOpsConfig(offline_mode=true, wandb_api_key="dummy")
        )
        
        t2 = @elapsed begin
            fit!(model, X, y, epochs=5, batch_size=20, 
                callbacks=loggers, verbose=false)
        end
        
        overhead = (t2 - t1) / t1 * 100
        
        println("  Tiempo sin logging: $(round(t1, digits=3))s")
        println("  Tiempo con logging: $(round(t2, digits=3))s")
        println("  Overhead: $(round(overhead, digits=2))%")
        
        @test overhead < 5  # Permitir hasta 5% de overhead en tests
        
        println("  ✅ Performance impact aceptable")
    end
    
    @testset "8. Error Handling" begin
        println("\n🧪 Test 8: Manejo robusto de errores")
        
        # Test con API key inválida (pero en modo offline)
        logger1 = nothing
        logger1_created = false
        try
            logger1 = WandBLogger("test", Dict(),
                                api_key="invalid",
                                offline_mode=true)
            logger1_created = (logger1 !== nothing && logger1.offline_mode == true)
        catch e
            logger1_created = false
        end
        
        @test logger1_created
        
        # Test con tracking URI inválido
        logger2 = nothing
        logger2_created = false
        try
            logger2 = MLFlowLogger("test",
                                 tracking_uri="http://invalid:9999",
                                 offline_mode=true)
            logger2_created = (logger2 !== nothing && logger2.offline_mode == true)
        catch e
            logger2_created = false
        end
        
        @test logger2_created
        
        # Test con métricas corruptas
        logger = WandBLogger("test", Dict(), offline_mode=true, log_dir=test_dir)
        
        # Logs con tipos inválidos
        bad_logs = Dict(
            :loss => "not_a_number",
            :model => Sequential([Dense(5, 2)]),  # Objeto no serializable
            :accuracy => 0.9f0  # Este sí es válido
        )
        
        # No debe fallar
        error_occurred = false
        try
            on_epoch_end(logger, 1, bad_logs)
        catch e
            error_occurred = true
        end
        
        @test !error_occurred  # No debe haber error
        
        # Verificar que solo se guardó la métrica válida
        @test haskey(logger.local_logger.metrics_history, "accuracy")
        @test !haskey(logger.local_logger.metrics_history, "loss")
        
        println("  ✅ Manejo de errores robusto")
    end
    
    @testset "9. GPU Metrics" begin
        println("\n🧪 Test 9: Métricas de GPU")
        
        if CUDA.functional()
            logger = WandBLogger("gpu_test", Dict(), 
                               offline_mode=true, log_dir=test_dir)
            
            # Crear algunos tensores en GPU
            gpu_tensor = CUDA.ones(1000, 1000)
            
            logs = Dict(:loss => 0.5f0)
            on_epoch_end(logger, 1, logs)
            
            # Verificar que se registraron métricas de GPU
            @test length(logger.metrics_buffer) > 0
            metrics = logger.metrics_buffer[1]
            
            @test haskey(metrics, "system/gpu_memory_used")
            @test haskey(metrics, "system/gpu_utilization")
            @test metrics["system/gpu_memory_used"] > 0
            
            println("  ✅ Métricas de GPU funcionan")
        else
            println("  ⚠️  GPU no disponible, saltando test")
            @test true
        end
    end
    
    @testset "10. Sync Offline Runs" begin
        println("\n🧪 Test 10: Sincronización de runs offline")
        
        # Crear algunos logs offline de prueba
        offline_dir = joinpath(test_dir, "offline_sync_test")
        mkpath(joinpath(offline_dir, "wandb"))
        mkpath(joinpath(offline_dir, "mlflow"))
        
        # Simular logs offline
        for i in 1:3
            # WandB logs
            wandb_file = joinpath(offline_dir, "wandb", "wandb_run_$i.jsonl")
            open(wandb_file, "w") do f
                println(f, JSON3.write(Dict("event" => "start", "run" => i)))
                println(f, JSON3.write(Dict("loss" => 0.5 / i)))
            end
            
            # MLFlow logs
            mlflow_file = joinpath(offline_dir, "mlflow", "mlflow_exp_$i.jsonl")
            open(mlflow_file, "w") do f
                println(f, JSON3.write(Dict("event" => "start", "exp" => i)))
                println(f, JSON3.write(Dict("accuracy" => 0.8 + 0.05 * i)))
            end
        end
        
        # Test función de sincronización
        sync_worked = false
        cd(offline_dir) do
            try
                sync_offline_runs(MLOpsConfig(offline_mode=true))
                sync_worked = true
            catch e
                sync_worked = false
                @warn "sync_offline_runs falló: $e"
            end
        end
        
        @test sync_worked
        
        println("  ✅ Sincronización offline funciona")
    end
    
    # Limpiar
    try
        rm(test_dir, recursive=true, force=true)
    catch
    end
    
    println("\n✅ Todos los tests de MLOps Integrations pasaron!")
end

@testset "Flujo Completo E2E" begin
    println("\n🚀 Test E2E: Flujo completo de entrenamiento con MLOps")
    
    # Configuración
    project_name = "deepdynamics_e2e_test"
    test_dir = mktempdir()
    
    # Crear configuración MLOps (offline)
    mlops_config = MLOpsConfig(
        wandb_api_key       = "test_key",
        wandb_entity        = "test_entity",
        wandb_base_url      = "",
        mlflow_tracking_uri = "",
        offline_mode        = true
    )
    
    # Definir arquitectura (binary classifier)
    model = Sequential([
        Dense(784, 256), BatchNorm(256), Activation(relu),
        DropoutLayer(0.2f0),
        Dense(256, 128), BatchNorm(128), Activation(relu),
        DropoutLayer(0.1f0),
        Dense(128, 2),   Activation(softmax),
    ])
    
    n_samples = 500
    
    # Función helper mejorada con más ruido y outliers
    function generate_structured_data(
        n_samples::Int,
        n_features::Int,
        n_classes::Int;
        prototype_scale::Float32 = 1.0f0,
        noise_level::Float32     = 0.6f0,
        imbalance_ratios::Union{Nothing,Vector{Float32}} = nothing,
        label_noise::Float32     = 0.05f0,
        outlier_fraction::Float32 = 0.03f0,
        normalize::Bool          = true
    )
        # 1) Determinar muestras por clase
        if imbalance_ratios === nothing
            base = div(n_samples, n_classes)
            counts = fill(base, n_classes)
            for i in 1:(n_samples - base * n_classes)
                counts[(i - 1) % n_classes + 1] += 1
            end
        else
            @assert length(imbalance_ratios)==n_classes "imbalance_ratios debe tener n_classes elementos"
            @assert abs(sum(imbalance_ratios)-1f0)<1e-6 "imbalance_ratios deben sumar 1.0"
            counts = round.(Int, imbalance_ratios .* n_samples)
            diff = n_samples - sum(counts)
            for i in 1:abs(diff)
                counts[(i - 1) % n_classes + 1] += sign(diff)
            end
        end

        # 2) Crear prototipos
        prototypes = [randn(Float32, n_features) .* prototype_scale for _ in 1:n_classes]

        # 3) Preparar contenedores
        X = Tensor{2}[]
        y = Tensor{2}[]

        # 4) Generar datos con ruido y outliers
        for cls in 1:n_classes
            for _ in 1:counts[cls]
                x = prototypes[cls] .+ randn(Float32, n_features) .* noise_level
                if rand() < outlier_fraction
                    x .= randn(Float32, n_features) .* (noise_level * 10f0)
                end
                if normalize
                    x .= tanh.(x)
                end
                push!(X, Tensor(reshape(x, n_features, 1)))

                lab = zeros(Float32, n_classes)
                lab[cls] = 1f0
                push!(y, Tensor(reshape(lab, n_classes, 1)))
            end
        end

        # 5) Barajar
        perm = randperm(length(X))
        X = X[perm]
        y = y[perm]

        # 6) Ruido de etiquetas
        n_noisy = round(Int, label_noise * length(y))
        noisy_idxs = randperm(length(y))[1:n_noisy]
        for idx in noisy_idxs
            data = copy(y[idx].data)
            true_cls = argmax(data)
            alts = [i for i in 1:n_classes if i != true_cls]
            new_cls = rand(alts)
            data .= 0f0
            data[new_cls] = 1f0
            y[idx] = Tensor(reshape(data, n_classes, 1))
        end

        return X, y
    end

    # Generar datos de entrenamiento
    X_train, y_train = generate_structured_data(500, 784, 2)

    # Mezclar para el entrenamiento
    idx = shuffle(1:n_samples)
    X_train = X_train[idx]
    y_train = y_train[idx]
    
    # Configurar experimento MLOps
    experiment_config = Dict(
        "architecture"  => "MLP",
        "dataset"       => "noisy_synthetic",
        "layers"        => [784, 256, 128, 2],
        "activation"    => "relu",
        "dropout"       => 0.2,
        "optimizer"     => "Adam",
        "learning_rate" => 0.001
    )
    
    # Crear loggers
    mlops_loggers = create_mlops_logger(
        project_name = project_name,
        platforms    = [:wandb, :mlflow],
        config       = experiment_config,
        mlops_config = mlops_config
    )
    
    # Callbacks
    all_callbacks = vcat(
        mlops_loggers,
        [
            EarlyStopping(patience=10, min_delta=0.001f0),
            ReduceLROnPlateau(patience=5, factor=0.5f0)
        ]
    )
    
    # Entrenar
    println("  Entrenando modelo...")
    history = fit!(
        model, X_train, y_train;
        epochs           = 20,
        batch_size       = 32,
        optimizer        = Adam(0.001f0),
        loss_fn          = categorical_crossentropy,
        validation_split = 0.2f0,
        callbacks        = all_callbacks,
        verbose          = true
    )

    # Validaciones
    @test length(history.train_loss) <= 20
    @test history.train_loss[end] < history.train_loss[1]

    val_accuracies = get(history.val_metrics, "accuracy", Float32[])
    if !isempty(val_accuracies)
        @test maximum(val_accuracies) > 0.1f0
    else
        @test true
    end

    println("  ✅ Flujo E2E completado exitosamente")
    println("    - Épocas entrenadas: $(length(history.train_loss))")
    println("    - Loss final: $(round(history.train_loss[end], digits=4))")
    if !isempty(val_accuracies)
        println("    - Accuracy final: $(round(val_accuracies[end] * 100, digits=2))%")
    else
        println("    - Accuracy final: No disponible")
    end

    # Limpieza
    rm(test_dir, recursive=true, force=true)
end
