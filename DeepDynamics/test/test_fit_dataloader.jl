# test_fit_dataloader_fixed.jl
using Test
using DeepDynamics
using Random
using CUDA

Random.seed!(42)

println("="^70)
println("TEST DE FIT! CON DATALOADER Y CALLBACKS")
println("="^70)

@testset "fit! con DataLoader" begin
    
    # Generar datos sintéticos
    function create_synthetic_data(n_samples=100, n_features=10, n_classes=3)
        X = [Tensor(randn(Float32, n_features)) for _ in 1:n_samples]
        y = []
        for _ in 1:n_samples
            label = zeros(Float32, n_classes)
            label[rand(1:n_classes)] = 1.0f0
            push!(y, Tensor(label))
        end
        return X, y
    end
    
    @testset "1. DataLoader básico" begin
        println("\n1️⃣ Test básico con DataLoader")
        
        X, y = create_synthetic_data(80, 8, 2)
        
        # Crear DataLoaders
        train_loader = DataLoader(X[1:60], y[1:60], 10, shuffle=true)
        val_loader = DataLoader(X[61:80], y[61:80], 10, shuffle=false)
        
        # Modelo con sintaxis correcta
        model = Sequential([
            Dense(8, 16),
            Activation(relu),
            Dense(16, 2),
            Activation(softmax)
        ])
        
        # Entrenar
        history = fit!(model, train_loader,
            val_loader=val_loader,
            epochs=5,
            optimizer=Adam(0.01f0),
            loss_fn=categorical_crossentropy,
            verbose=1
        )
        
        @test length(history.train_loss) == 5
        @test length(history.val_loss) == 5
        @test history.train_loss[end] < history.train_loss[1]
        
        println("  ✅ DataLoader básico funcionando")
    end
    
    @testset "2. Early Stopping" begin
        println("\n2️⃣ Test de Early Stopping")
        
        X, y = create_synthetic_data(100, 10, 3)
        train_loader = DataLoader(X, y, 20, shuffle=true)
        
        model = Sequential([
            Dense(10, 20),
            Activation(relu),
            Dense(20, 3)
        ])
        
        # CORREGIDO: usar Float32 para min_delta
        early_stop = EarlyStopping(patience=2, min_delta=0.1f0)
        
        history = fit!(model, train_loader,
            epochs=50,
            callbacks=[early_stop],
            verbose=1
        )
        
        @test history.epochs < 50
        @test early_stop.stopped
        
        println("  ✅ Early stopping activado en época $(history.epochs)")
    end
    
    @testset "3. ModelCheckpoint" begin
        println("\n3️⃣ Test de ModelCheckpoint")
        
        X, y = create_synthetic_data(80, 6, 2)
        train_loader = DataLoader(X[1:60], y[1:60], 12)
        val_loader = DataLoader(X[61:80], y[61:80], 12)
        
        model = Sequential([
            Dense(6, 12),
            BatchNorm(12),
            Activation(relu),
            Dense(12, 2)
        ])
        
        # CORREGIDO: usar String para monitor
        checkpoint = ModelCheckpoint(
            "test_model.jld2",
            monitor="val_loss",
            mode=:min,
            save_best_only=true,
            verbose=true
        )
        
        history = fit!(model, train_loader,
            val_loader=val_loader,
            epochs=10,
            callbacks=[checkpoint],
            verbose=1
        )
        
        @test checkpoint.best< Inf
        
        println("  ✅ ModelCheckpoint registró mejor valor: $(checkpoint.best)")
    end
    
    @testset "4. ReduceLROnPlateau" begin
        println("\n4️⃣ Test de ReduceLROnPlateau")
        
        X, y = create_synthetic_data(100, 8, 2)
        train_loader = DataLoader(X[1:80], y[1:80], 16)
        val_loader = DataLoader(X[81:100], y[81:100], 16)
        
        model = Sequential([
            Dense(8, 16),
            Activation(relu),
            Dense(16, 2)
        ])
        
        opt = Adam(0.01f0)
        initial_lr = opt.learning_rate
        
        # CORREGIDO: usar String para monitor
        reduce_lr = ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=true
        )
        
        history = fit!(model, train_loader,
            val_loader=val_loader,
            epochs=20,
            optimizer=opt,
            callbacks=[reduce_lr],
            verbose=1
        )
        
        final_lr = opt.learning_rate
        @test final_lr <= initial_lr
        
        println("  ✅ LR reducido: $initial_lr → $final_lr")
    end
    
    @testset "5. Progress bar (verbose=2)" begin
        println("\n5️⃣ Test de progress bar detallado")
        
        X, y = create_synthetic_data(50, 5, 2)
        train_loader = DataLoader(X, y, 5)
        
        model = Sequential([
            Dense(5, 10),
            Activation(relu),
            Dense(10, 2)
        ])
        
        println("  Entrenando con progress bar por batch:")
        history = fit!(model, train_loader,
            epochs=2,
            verbose=2  # Progress bar detallado
        )
        
        @test length(history.train_loss) == 2
        println("\n  ✅ Progress bar funcionó correctamente")
    end
    
    @testset "6. Múltiples callbacks" begin
        println("\n6️⃣ Test con múltiples callbacks")
        
        X, y = create_synthetic_data(120, 10, 4)
        train_loader = DataLoader(X[1:100], y[1:100], 20)
        val_loader = DataLoader(X[101:120], y[101:120], 20)
        
        model = Sequential([
            Dense(10, 30),
            DropoutLayer(0.2),
            Activation(relu),
            Dense(30, 15),
            BatchNorm(15),
            Activation(relu),
            Dense(15, 4)
        ])
        
        opt = Adam(0.01f0)
        
        # CORREGIDO: tipos correctos para todos los callbacks
        callbacks = [
            EarlyStopping(patience=5, min_delta=0.001f0),  # Float32
            ReduceLROnPlateau(monitor="val_loss", factor=0.7, patience=3),  # String
            ModelCheckpoint("best_model_multi.jld2", monitor="val_loss")  # String
        ]
        
        history = fit!(model, train_loader,
            val_loader=val_loader,
            epochs=30,
            optimizer=opt,
            loss_fn=categorical_crossentropy,
            callbacks=callbacks,
            verbose=1
        )
        
        @test length(history.train_loss) > 0
        @test length(history.val_loss) > 0
        @test haskey(history.train_metrics, "accuracy")
        
        println("  ✅ Múltiples callbacks funcionando juntos")
    end
    
    @testset "7. GPU compatibility" begin
        println("\n7️⃣ Test de compatibilidad GPU")
        
        X, y = create_synthetic_data(40, 6, 3)
        train_loader = DataLoader(X, y, 8)
        
        model = Sequential([
            Dense(6, 12),
            Activation(relu),
            Dense(12, 3)
        ])
        
        if CUDA.functional()
            model = model_to_gpu(model)
            println("  🖥️  Probando en GPU")
        else
            println("  💻 Probando en CPU")
        end
        
        history = fit!(model, train_loader,
            epochs=3,
            verbose=1
        )
        
        @test length(history.train_loss) == 3
        @test all(isfinite.(history.train_loss))
        
        println("  ✅ Compatibilidad GPU/CPU verificada")
    end
    
    @testset "8. Optimized DataLoader" begin
        println("\n8️⃣ Test con OptimizedDataLoader")
        
        X, y = create_synthetic_data(100, 8, 2)
        
        # Crear data loader optimizado
        train_loader = optimized_data_loader(X[1:80], y[1:80], 16,
            shuffle=true, to_gpu=CUDA.functional(), prefetch=2)
        val_loader = optimized_data_loader(X[81:100], y[81:100], 16,
            shuffle=false, to_gpu=CUDA.functional(), prefetch=1)
        
        model = Sequential([
            Dense(8, 16),
            Activation(relu),
            Dense(16, 2)
        ])
        
        if CUDA.functional()
            model = model_to_gpu(model)
        end
        
        history = fit!(model, train_loader,
            val_loader=val_loader,
            epochs=5,
            verbose=1
        )
        
        @test length(history.train_loss) == 5
        
        # Limpiar data loaders
        cleanup_data_loader!(train_loader)
        cleanup_data_loader!(val_loader)
        
        println("  ✅ OptimizedDataLoader funcionando")
    end
end

# Test de compatibilidad con fit! original
@testset "Compatibilidad con fit! original" begin
    println("\n🔄 Test de compatibilidad con fit! para arrays")
    
    X = [Tensor(randn(Float32, 5, 1)) for _ in 1:50]
    y = [Tensor(Float32.([i % 2, 1 - i % 2])) for i in 1:50]
    
    model = Sequential([
        Dense(5, 10),
        Activation(relu),
        Dense(10, 2)
    ])
    
    # Usar fit! con arrays (función original)
    history = fit!(model, X, y,
        epochs=3,
        batch_size=10,
        validation_split=0.2f0,
        verbose=true
    )
    
    @test length(history.train_loss) == 3
    @test length(history.val_loss) == 3
    
    println("  ✅ fit! original sigue funcionando")
end

println("\n" * "="^70)
println("✨ TODOS LOS TESTS COMPLETADOS CON ÉXITO ✨")
println("="^70)