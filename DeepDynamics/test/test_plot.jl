# test/test_visualizations.jl
using Test
using DeepDynamics
using DeepDynamics.Visualizations
using DeepDynamics.Callbacks
using Statistics

println("="^60)
println("Test de Visualizations.jl")
println("="^60)

@testset "Visualizations Tests" begin
    
    @testset "1. Funciones básicas de plotting" begin
        println("\n1️⃣ Test funciones básicas de plotting")
        
        # Suprimir output gráfico en tests
        ENV["GKSwstype"] = "100"
        
        # Test plot_training_progress
        train_losses = [1.0, 0.8, 0.6, 0.4, 0.2]
        val_losses = [1.1, 0.9, 0.7, 0.5, 0.3]
        
        @test_nowarn plot_training_progress(train_losses, val_losses)
        @test_nowarn plot_training_progress(train_losses, nothing)
        
        # Test plot_metrics
        @test_nowarn plot_metrics(train_losses, val_losses)
        
        println("  ✅ Funciones básicas funcionando")
    end
    
    @testset "2. plot_training_history" begin
        println("\n2️⃣ Test plot_training_history")
        
        # Crear historia de ejemplo
        history = Dict(
            :train_loss => [1.0, 0.8, 0.6, 0.4, 0.2],
            :val_loss => [1.1, 0.9, 0.7, 0.5, 0.3],
            :train_accuracy => [0.5, 0.6, 0.7, 0.8, 0.9],
            :val_accuracy => [0.45, 0.55, 0.65, 0.75, 0.85]
        )
        
        # Test sin guardar
        @test_nowarn plot_training_history(history)
        
        # Test con diferentes configuraciones
        @test_nowarn plot_training_history(history; figsize=(10, 6))
        
        # Test guardando archivo
        mktempdir() do tmpdir
            save_path = joinpath(tmpdir, "test_plot.png")
            @test_nowarn plot_training_history(history; save_path=save_path)
            @test isfile(save_path)
        end
        
        # Test con solo pérdidas
        history_minimal = Dict(
            :train_loss => [1.0, 0.8, 0.6],
            :val_loss => [1.1, 0.9, 0.7]
        )
        @test_nowarn plot_training_history(history_minimal)
        
        println("  ✅ plot_training_history funcionando")
    end
    
    @testset "3. moving_average" begin
        println("\n3️⃣ Test moving_average")
        
        # Test básico
        data = Float64[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        ma = Visualizations.moving_average(data, 3)
        
        @test length(ma) == 8  # 10 - 3 + 1
        @test ma[1] ≈ mean([1, 2, 3])
        @test ma[2] ≈ mean([2, 3, 4])
        @test ma[end] ≈ mean([8, 9, 10])
        
        # Test con ventana de 1
        ma1 = Visualizations.moving_average(data, 1)
        @test ma1 == data
        
        # Test con ventana igual al tamaño
        ma_full = Visualizations.moving_average(data, 10)
        @test length(ma_full) == 1
        @test ma_full[1] ≈ mean(data)
        
        # Test con ventana mayor que datos
        ma_big = Visualizations.moving_average(data, 20)
        @test length(ma_big) == 1
        @test ma_big[1] ≈ mean(data)
        
        println("  ✅ moving_average funcionando correctamente")
    end
    
    @testset "4. LivePlotter callback" begin
        println("\n4️⃣ Test LivePlotter callback")
        
        # Crear LivePlotter
        plotter = LivePlotter(update_freq=2, metrics=["accuracy", "loss"])
        
        # Verificar tipo
        @test plotter isa AbstractCallback
        @test plotter.update_freq == 2
        @test plotter.metrics == ["accuracy", "loss"]
        
        # Test on_train_begin
        logs = Dict{Symbol, Any}()
        @test_nowarn on_train_begin(plotter, logs)
        @test plotter.batch_count == 0
        @test isempty(plotter.loss_history)
        @test haskey(plotter.metric_history, "accuracy")
        @test haskey(plotter.metric_history, "loss")
        
        # Test on_batch_end
        batch_logs = Dict(:loss => 0.5, :accuracy => 0.8)
        @test_nowarn on_batch_end(plotter, 1, batch_logs)
        @test plotter.batch_count == 1
        @test plotter.loss_history == [0.5]
        @test plotter.metric_history["accuracy"] == [0.8]
        
        # Test múltiples batches
        for i in 2:10
            batch_logs = Dict(:loss => 0.5 - i*0.01, :accuracy => 0.8 + i*0.01)
            @test_nowarn on_batch_end(plotter, i, batch_logs)
        end
        
        @test length(plotter.loss_history) == 10
        @test plotter.batch_count == 10
        
        # Test on_epoch_end
        epoch_logs = Dict(:loss => 0.3, :val_loss => 0.35)
        @test_nowarn on_epoch_end(plotter, 1, epoch_logs)
        
        # Test on_train_end
        @test_nowarn on_train_end(plotter, Dict())
        
        println("  ✅ LivePlotter callback funcionando")
    end
    
    @testset "5. plot_model_architecture" begin
        println("\n5️⃣ Test plot_model_architecture")
        
        # Crear modelo de prueba usando Activation en lugar de funciones lambda
        model = Sequential([
            Dense(10, 20),
            Activation(relu),
            Dense(20, 20),
            DropoutLayer(0.5),
            Dense(20, 3),
            Activation(softmax)
        ])
        
        # Test básico
        @test_nowarn plot_model_architecture(model)
        
        # Test sin mostrar parámetros
        @test_nowarn plot_model_architecture(model; show_params=false)
        
        # Test guardando archivo
        mktempdir() do tmpdir
            save_path = joinpath(tmpdir, "architecture.png")
            @test_nowarn plot_model_architecture(model; save_path=save_path)
            @test isfile(save_path)
        end
        
        # Test con modelo CNN
        cnn_model = Sequential([
            Conv2D(3, 32, (3,3)),
            Activation(relu),
            MaxPooling((2,2)),
            BatchNorm(32),
            Flatten(),
            Dense(100, 10)
        ])
        
        @test_nowarn plot_model_architecture(cnn_model)
        
        println("  ✅ plot_model_architecture funcionando")
    end
    
    @testset "6. extract_layer_info" begin
        println("\n6️⃣ Test extract_layer_info")
        
        # Modelo simple con Activation en lugar de lambda
        model = Sequential([
            Dense(10, 20),
            Activation(relu),
            DropoutLayer(0.5)
        ])
        
        layer_info = Visualizations.extract_layer_info(model)
        
        @test length(layer_info) == 3
        @test layer_info[1][:type] == "Dense"
        @test layer_info[1][:params] == 10*20 + 20  # weights + bias
        @test layer_info[2][:type] == "Activation"
        @test layer_info[2][:name] == "ReLU"
        @test layer_info[3][:type] == "Dropout"
        
        println("  ✅ extract_layer_info funcionando")
    end
    
    @testset "7. Funciones auxiliares" begin
        println("\n7️⃣ Test funciones auxiliares")
        
        # Test get_layer_color
        @test Visualizations.get_layer_color("Dense") == :lightblue
        @test Visualizations.get_layer_color("Conv2D") == :lightgreen
        @test Visualizations.get_layer_color("Unknown") == :white
        @test Visualizations.get_layer_color("CustomLayer") == :white
        
        # Test format_params
        @test Visualizations.format_params(100) == "100"
        @test Visualizations.format_params(1_500) == "1.5K"
        @test Visualizations.format_params(2_500_000) == "2.5M"
        
        println("  ✅ Funciones auxiliares funcionando")
    end
    
    @testset "8. Integración con fit!" begin
        println("\n8️⃣ Test integración con sistema de entrenamiento")
        
        # Datos sintéticos
        X = [Tensor(randn(Float32, 5)) for _ in 1:50]
        y = [Tensor(randn(Float32, 2)) for _ in 1:50]
        
        model = Sequential([
            Dense(5, 10),
            Activation(relu),
            Dense(10, 2)
        ])
        
        # Crear LivePlotter
        plotter = LivePlotter(update_freq=5, metrics=["loss"])
        
        # Entrenar con LivePlotter
        history = fit!(model, X, y,
            epochs=3,
            batch_size=10,
            callbacks=[plotter],
            verbose=false
        )
        
        # Verificar que se actualizó
        @test plotter.batch_count > 0
        @test !isempty(plotter.loss_history)
        
        # Visualizar historia final
        history_dict = Dict(
            :train_loss => history.train_loss,
            :val_loss => history.val_loss
        )
        @test_nowarn plot_training_history(history_dict)
        
        println("  ✅ Integración con fit! funcionando")
    end
    
    @testset "9. plot_conv_filters" begin
        println("\n9️⃣ Test plot_conv_filters")
        
        # Crear filtros de prueba
        filters = randn(3, 3, 1, 8)  # 8 filtros de 3x3
        
        @test_nowarn plot_conv_filters(filters)
        
        # Test con más filtros
        many_filters = randn(5, 5, 3, 16)  # 16 filtros de 5x5 con 3 canales
        @test_nowarn plot_conv_filters(many_filters)
        
        println("  ✅ plot_conv_filters funcionando")
    end
    
    @testset "10. Manejo de memoria" begin
        println("\n🔟 Test manejo de memoria en LivePlotter")
        
        plotter = LivePlotter(update_freq=10)
        
        # Simular muchos batches
        for i in 1:100
            batch_logs = Dict(:loss => rand())
            on_batch_end(plotter, i, batch_logs)
        end
        
        # Verificar funcionamiento
        @test plotter.batch_count == 100
        @test length(plotter.loss_history) == 100
        
        println("  ✅ Manejo de memoria funcionando")
    end
end

println("\n✅ Todos los tests de Visualizations pasaron!")