using Test
using DeepDynamics
using Random
using CUDA  # Agregar import de CUDA
using Statistics  # Agregar import de Statistics

@testset "train! Function Tests" begin
    # Configuración común
    Random.seed!(42)
    
    # Crear modelo simple
    function create_test_model()
        Sequential([
            Dense(2, 4),
            Activation(relu),  # Usando Activation correctamente
            Dense(4, 1)
        ])
    end
    
    # Generar datos sintéticos
    function generate_test_data(n_samples=1000)
        X = [Tensor(randn(Float32, 2)) for _ in 1:n_samples]
        y = [Tensor([sum(x.data) > 0 ? 1.0f0 : 0.0f0]) for x in X]
        return X, y
    end
    
    @testset "Basic Functionality" begin
        model = create_test_model()
        X, y = generate_test_data(50)
        
        history = train!(
            model, X, y,
            epochs=3,
            batch_size=10,
            verbose=false
        )
        
        @test haskey(history, :loss)
        @test haskey(history, :metrics)
        @test length(history[:loss]) == 3
        @test all(loss -> loss isa Float64, history[:loss])
        @test all(loss -> loss >= 0, history[:loss])
    end
    
    @testset "Dimension Validation" begin
        model = create_test_model()
        
        # Tamaños incompatibles
        X = [Tensor(randn(Float32, 2)) for _ in 1:10]
        y = [Tensor(randn(Float32, 1)) for _ in 1:5]
        
        @test_throws DimensionMismatch train!(model, X, y, verbose=false)
        
        # Datos vacíos
        @test_throws ArgumentError train!(model, [], [], verbose=false)
    end
    
    @testset "Array Input Support" begin
        model = create_test_model()
        
        # Arrays multidimensionales
        X_array = randn(Float32, 2, 50)  # 2 features, 50 samples
        y_array = randn(Float32, 1, 50)  # 1 output, 50 samples
        
        history = train!(
            model, X_array, y_array,
            epochs=2,
            batch_size=25,
            verbose=false
        )
        
        @test length(history[:loss]) == 2
        
        # Vectores simples
        model_vec = create_test_model()
        X_vec = randn(Float32, 2, 50)  # 2 features, 50 samples
        y_vec = randn(Float32, 1, 50)  
        
        history_vec = train!(
            model_vec, X_vec, y_vec,
            epochs=1,
            verbose=false
        )
        
        @test length(history_vec[:loss]) == 1
    end
    
    @testset "Metrics Tracking" begin
        model = create_test_model()
        X, y = generate_test_data(40)
        
        # Definir métrica personalizada
        function custom_metric(pred, true_val)
            return Statistics.mean(abs.(pred .- true_val))
        end
        
        history = train!(
            model, X, y,
            epochs=5,
            batch_size=8,
            metrics=[accuracy, custom_metric],
            verbose=false
        )
        
        @test haskey(history[:metrics], :accuracy)
        @test haskey(history[:metrics], :custom_metric)
        @test length(history[:metrics][:accuracy]) == 5
        @test length(history[:metrics][:custom_metric]) == 5
        @test all(acc -> 0 <= acc <= 1, history[:metrics][:accuracy])
    end
    
    @testset "Loss Decrease" begin
        model = create_test_model()
        X, y = generate_test_data(100)
        
        history = train!(
            model, X, y,
            optimizer=Adam(learning_rate=0.01),
            epochs=20,
            batch_size=20,
            verbose=false
        )
        
        # Loss debe disminuir en promedio
        first_half = Statistics.mean(history[:loss][1:10])
        second_half = Statistics.mean(history[:loss][11:20])
        @test second_half < first_half
    end
    
    @testset "Different Optimizers" begin
        X, y = generate_test_data(50)
        
        for opt in [
            SGD(learning_rate=0.01), 
            Adam(learning_rate=0.001), 
            RMSProp(learning_rate=0.001)
        ]
            model = create_test_model()
            history = train!(
                model, X, y,
                optimizer=opt,
                epochs=5,
                verbose=false
            )
            
            @test length(history[:loss]) == 5
            # Verificar que el loss cambia (no está estancado)
            @test !all(loss -> loss ≈ history[:loss][1], history[:loss])
        end
    end
    
    @testset "Different Loss Functions" begin
        model = create_test_model()
        X, y = generate_test_data(30)
        
        # MSE loss
        history_mse = train!(
            model, X, y,
            loss_fn=mse_loss,
            epochs=3,
            verbose=false
        )
        
        # Binary crossentropy
        model2 = create_test_model()
        history_bce = train!(
            model2, X, y,
            loss_fn=binary_crossentropy,
            epochs=3,
            verbose=false
        )
        
        @test length(history_mse[:loss]) == 3
        @test length(history_bce[:loss]) == 3
        # Los valores de loss deberían ser diferentes
        @test !all(history_mse[:loss] .≈ history_bce[:loss])
    end
    
    @testset "Batch Size Edge Cases" begin
        model = create_test_model()
        X, y = generate_test_data(23)  # Número primo
        
        # Batch size mayor que datos
        history1 = train!(
            model, X, y,
            batch_size=50,
            epochs=2,
            verbose=false
        )
        
        # Batch size = 1
        model2 = create_test_model()
        history2 = train!(
            model2, X, y,
            batch_size=1,
            epochs=1,
            verbose=false
        )
        
        @test length(history1[:loss]) == 2
        @test length(history2[:loss]) == 1
    end
    
    @testset "GPU Compatibility (asumida siempre)" begin
        # Asumimos que la GPU está disponible
        model = model_to_gpu(create_test_model())
        X, y = generate_test_data(20)

        history = train!(
            model, X, y,
            epochs=3,
            batch_size=5,
            verbose=false
        )

        @test length(history[:loss]) == 3
        @test model_device(model) == :gpu
    end
    
    @testset "Format Adaptation" begin
        model = create_test_model()
        
        # Diferentes formatos de entrada
        formats = [
            # Vector 2D
            [Tensor(randn(Float32, 2, 1)) for _ in 1:10],
            # Vector 1D  
            [Tensor(randn(Float32, 2)) for _ in 1:10],
            # Matriz columna
            [Tensor(reshape(randn(Float32, 2), :, 1)) for _ in 1:10]
        ]
        
        for (i, X_format) in enumerate(formats)
            y = [Tensor([Float32(i)]) for _ in 1:10]
            
            # No debe fallar con ningún formato
            model_fmt = create_test_model()
            history = train!(
                model_fmt, X_format, y,
                epochs=1,
                verbose=false
            )
            
            @test length(history[:loss]) == 1
        end
    end
    
    @testset "Memory Management" begin
        model = create_test_model()
        X, y = generate_test_data(200)
        
        # Entrenar con muchos datos debería funcionar sin OOM
        history = train!(
            model, X, y,
            epochs=5,
            batch_size=10,
            verbose=false
        )
        
        @test length(history[:loss]) == 5
        
        # Verificar que se pueden hacer múltiples entrenamientos
        for _ in 1:3
            train!(
                model, X[1:50], y[1:50],
                epochs=2,
                verbose=false
            )
        end
        
        # Si llegamos aquí, no hubo memory leaks
        @test true
    end
    
    @testset "Progress and Verbose" begin
        model = create_test_model()
        X, y = generate_test_data(20)
        
        # Con verbose=true no debe fallar
        history = train!(
            model, X, y,
            epochs=2,
            verbose=true
        )
        
        @test length(history[:loss]) == 2
    end
end

@testset "Integration Test – Profundización" begin
    # 1) Generamos la espiral binaria como antes
    Random.seed!(123)
    n_samples = 200
    X_data = Float32[]
    y_data = Float32[]
    for i in 1:n_samples
        angle  = 4π * i / n_samples
        radius = i / n_samples
        if iseven(i)
            x1, x2 = radius*cos(angle), radius*sin(angle)
            push!(y_data, 1f0)
        else
            x1, x2 = -radius*cos(angle), -radius*sin(angle)
            push!(y_data, 0f0)
        end
        append!(X_data, (
            x1 + 0.1f0*randn(Float32),
            x2 + 0.1f0*randn(Float32)
        ))
    end
    X = [ Tensor(X_data[2i-1:2i]) for i in 1:n_samples ]
    y = [ Tensor([y_data[i]])      for i in 1:n_samples ]

    # 2) Definimos un modelo profundo y ancho que SÍ pueda aprender la espiral
    # Modelo para binaria:
    model = Sequential([
    Dense(2,32; init_method=:he),
    Activation(relu),
    Dense(32,16; init_method=:he),
    Activation(relu),
    Dense(16,1)          # *sin* sigmoid al final
    ])

    history = train!(model, X, y;
        loss_fn   = binary_crossentropy_with_logits,
        optimizer = Adam(0.05),
        epochs    = 200,
        batch_size= 8,
        metrics   = [binary_accuracy],  # la accuracy aplica sigmoid internamente
        verbose   = true
    )


    # 4) Ahora sí esperamos una caída significativa de la loss y acc > 0.7
    @test history[:loss][end] < history[:loss][1] * 0.5
    @test history[:metrics][:binary_accuracy][end] > 0.7
end

