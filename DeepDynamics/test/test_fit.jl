# test_fit_function.jl
# Test completo para la nueva función fit!

using Test
using DeepDynamics
using Random
using Statistics

Random.seed!(42)  # Para reproducibilidad

println("="^70)
println("TEST DE LA FUNCIÓN FIT!")
println("="^70)

@testset "fit! function tests" begin
    
    # Generar datos sintéticos de clasificación
    function generate_synthetic_data(n_samples=100, n_features=10, n_classes=3)
        # Generar características aleatorias
        X = [Tensor(randn(Float32, n_features, 1)) for _ in 1:n_samples]
        
        # Generar etiquetas one-hot - ESPECIFICAR TIPO
        y = Tensor[]  # Importante: tipo específico
        for i in 1:n_samples
            label = zeros(Float32, n_classes, 1)
            class_idx = rand(1:n_classes)
            label[class_idx, 1] = 1.0f0
            push!(y, Tensor(label))
        end
        
        return X, y
    end
    
    @testset "1. Funcionamiento básico" begin
        println("\n1️⃣ Test básico de fit!")
        
        # Datos pequeños
        X, y = generate_synthetic_data(50, 5, 2)
        
        # Modelo simple
        model = Sequential([
            Dense(5, 10),
            Activation(relu),
            Dense(10, 2),
            Activation(softmax)
        ])
        
        # Entrenar
        history = fit!(model, X, y,
            epochs=5,
            batch_size=10,
            validation_split=0.0f0,  # Sin validación por ahora
            verbose=true  # Bool, no Int
        )
        
        # Verificar historia
        @test length(history.train_loss) == 5
        @test all(history.train_loss .> 0)
        @test history.epochs == 5
        
        # La pérdida debería disminuir
        @test history.train_loss[end] < history.train_loss[1]
        
        println("  ✅ Entrenamiento básico completado")
    end
    
    @testset "2. Auto-split de validación" begin
        println("\n2️⃣ Test de auto-split")
        
        X, y = generate_synthetic_data(100, 8, 3)
        
        model = Sequential([
            Dense(8, 16),
            Activation(relu),
            Dense(16, 3),
            Activation(softmax)
        ])
        
        # Entrenar con auto-split
        history = fit!(model, X, y,
            epochs=10,
            batch_size=20,
            validation_split=0.2f0,
            optimizer=Adam(0.01f0),
            loss_fn=categorical_crossentropy,
            verbose=true  # Bool
        )
        
        # Verificar que hay datos de validación
        @test length(history.val_loss) == 10
        @test all(history.val_loss .> 0)
        
        # Verificar que train y val tienen diferente comportamiento
        @test history.train_loss != history.val_loss
        
        println("  ✅ Auto-split funcionando correctamente")
    end
    
    @testset "3. Validación explícita" begin
        println("\n3️⃣ Test con validación explícita")
        
        # Datos de entrenamiento y validación separados
        X_train, y_train = generate_synthetic_data(80, 10, 4)
        X_val, y_val = generate_synthetic_data(20, 10, 4)
        
        model = Sequential([
            Dense(10, 20),
            BatchNorm(20),
            Activation(relu),
            Dense(20, 4)
        ])
        
        history = fit!(model, X_train, y_train,
            X_val=X_val, y_val=y_val,
            epochs=8,
            batch_size=16,
            optimizer=SGD(learning_rate=0.1f0),
            loss_fn=categorical_crossentropy,
            verbose=true  # Bool
        )
        
        @test length(history.train_loss) == 8
        @test length(history.val_loss) == 8
        @test history.train_loss[end] < history.train_loss[1]
        
        println("  ✅ Validación explícita funcionando")
    end
    
    @testset "4. Métricas adicionales" begin
        println("\n4️⃣ Test de métricas")
        
        X, y = generate_synthetic_data(60, 6, 2)
        
        model = Sequential([
            Dense(6, 12),
            Activation(relu),
            Dense(12, 2),
            Activation(softmax)
        ])
        
        history = fit!(model, X, y,
            epochs=5,
            batch_size=12,
            validation_split=0.2f0,
            metrics=[:accuracy],
            verbose=true # Bool
        )
        
        # Verificar que las métricas se registraron
        @test haskey(history.train_metrics, "accuracy")
        @test haskey(history.val_metrics, "accuracy")
        @test length(history.train_metrics["accuracy"]) == 5
        @test length(history.val_metrics["accuracy"]) == 5
        
        # La precisión debería mejorar
        train_acc = history.train_metrics["accuracy"]
        @test train_acc[end] >= train_acc[1]
        
        println("  ✅ Métricas calculadas correctamente")
    end
    
    @testset "5. Early Stopping" begin
        println("\n5️⃣ Test de Early Stopping")
        
        X, y = generate_synthetic_data(100, 10, 3)
        
        model = Sequential([
            Dense(10, 50),
            Activation(relu),
            Dense(50, 50),
            Activation(relu),
            Dense(50, 3)
        ])
        
        # Crear callback de early stopping
        es = EarlyStopping(patience=3, min_delta=0.001f0)  # Float32
        
        history = fit!(model, X, y,
            epochs=50,  # Muchas épocas, pero debería parar antes
            batch_size=20,
            validation_split=0.2f0,
            callbacks=[es],
            verbose=true
        )
        
        # Debería haber parado antes de 50 épocas
        @test history.epochs < 50
        @test length(history.train_loss) < 50
        
        println("  ✅ Early stopping funcionó (paró en época $(history.epochs))")
    end
    
    @testset "6. Callbacks personalizados" begin
        println("\n6️⃣ Test de callbacks personalizados")
        
        # Crear un callback personalizado para contar épocas
        mutable struct EpochCounter <: AbstractCallback  # Corregido
            count::Int
            train_begin_called::Bool
            train_end_called::Bool
            
            EpochCounter() = new(0, false, false)
        end
        
        function DeepDynamics.Callbacks.on_epoch_end(cb::EpochCounter, epoch::Int, logs::Dict)
            cb.count += 1
        end
        
        function DeepDynamics.Callbacks.on_train_begin(cb::EpochCounter, logs::Dict)
            cb.train_begin_called = true
        end
        
        function DeepDynamics.Callbacks.on_train_end(cb::EpochCounter, logs::Dict)
            cb.train_end_called = true
        end
        
        X, y = generate_synthetic_data(40, 5, 2)
        
        model = Sequential([
            Dense(5, 10),
            Activation(relu),
            Dense(10, 2)
        ])
        
        counter = EpochCounter()
        
        history = fit!(model, X, y,
            epochs=7,
            batch_size=10,
            callbacks=[counter],
            verbose=true # Bool
        )
        
        @test counter.count == 7
        @test counter.train_begin_called
        @test counter.train_end_called
        
        println("  ✅ Callbacks personalizados funcionando")
    end
    
    @testset "7. Diferentes optimizadores" begin
        println("\n7️⃣ Test con diferentes optimizadores")
        
        X, y = generate_synthetic_data(50, 8, 2)
        
        optimizers = [
            ("SGD", SGD(learning_rate=0.1f0)),
            ("Adam", Adam(0.01f0)),
            ("RMSProp", RMSProp(learning_rate=0.01f0))  # Corregido
        ]
        
        for (name, opt) in optimizers
            model = Sequential([
                Dense(8, 16),
                Activation(relu),
                Dense(16, 2)
            ])
            
            history = fit!(model, X, y,
                epochs=5,
                batch_size=10,
                optimizer=opt,
                loss_fn=mse_loss,
                verbose=true # Bool
            )
            
            @test length(history.train_loss) == 5
            @test history.train_loss[end] < history.train_loss[1]
            
            println("  ✅ $name funcionó correctamente")
        end
    end
    
    @testset "8. Verbosidad" begin
        println("\n8️⃣ Test de niveles de verbosidad")
        
        X, y = generate_synthetic_data(30, 4, 2)
        
        model = Sequential([
            Dense(4, 8),
            Activation(relu),
            Dense(8, 2)
        ])
        
        # verbose=false (silencioso)
        println("  Probando verbose=false (no debería imprimir nada):")
        history = fit!(model, X, y, epochs=2, batch_size=10, verbose=false)
        @test length(history.train_loss) == 2
        
        # verbose=true (progreso por época)
        println("  Probando verbose=true (progreso por época):")
        history = fit!(model, X, y, epochs=2, batch_size=10, verbose=true)
        @test length(history.train_loss) == 2
        
        println("  ✅ Niveles de verbosidad funcionando")
    end
    
    @testset "9. GPU/CPU compatibility" begin
        println("\n9️⃣ Test de compatibilidad GPU/CPU")
        
        X, y = generate_synthetic_data(40, 6, 3)
        
        # Modelo que funcionará en CPU o GPU según disponibilidad
        model = Sequential([
            Dense(6, 12),
            Activation(relu),
            Dense(12, 3)
        ])
        
        # Si GPU está disponible, mover modelo
        if DeepDynamics.gpu_available()
            model = model_to_gpu(model)
            println("  🖥️  Probando en GPU")
        else
            println("  💻 Probando en CPU")
        end
        
        history = fit!(model, X, y,
            epochs=5,
            batch_size=8,
            validation_split=0.2f0,
            verbose=true  # Bool
        )
        
        @test length(history.train_loss) == 5
        @test all(isfinite.(history.train_loss))
        
        println("  ✅ Compatibilidad GPU/CPU verificada")
    end


    @testset "10. Reduce LR on Plateau" begin
        println("\n🔟 Test de ReduceLROnPlateau")
        
        X, y = generate_synthetic_data(80, 10, 2)
        
        model = Sequential([
            Dense(10, 20),
            Activation(relu),
            Dense(20, 2)
        ])
        
        # Optimizador con learning rate inicial
        opt = Adam(0.01f0)
        initial_lr = opt.learning_rate
        
        # Callback para reducir LR
        reduce_lr = ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5f0,  # Corregido el typo
            patience=2,
            min_lr=1e-6f0
        )
        
        history = fit!(model, X, y,
            epochs=20,
            batch_size=16,
            validation_split=0.2f0,
            optimizer=opt,
            callbacks=[reduce_lr],
            verbose=true # Bool
        )
        
        # El learning rate debería haber cambiado si no hubo mejora
        final_lr = opt.learning_rate
        @test final_lr <= initial_lr
        
        println("  ✅ ReduceLROnPlateau funcionando (LR: $initial_lr → $final_lr)")
    end
end

# Resumen final
println("\n" * "="^70)
println("✨ TODOS LOS TESTS DE FIT! COMPLETADOS ✨")
println("="^70)