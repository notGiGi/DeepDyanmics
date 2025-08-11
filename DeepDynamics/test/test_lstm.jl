using Test
using DeepDynamics
using CUDA
using Statistics

@testset "LSTM/GRU Complete Tests" begin
    
    # ========================================
    # TESTS BÁSICOS
    # ========================================
    @testset "1. Forward Pass Shapes" begin
        println("\n1️⃣ Test de shapes en forward pass")
        
        lstm_seq = LSTM(10, 20; return_sequences=true)
        x = Tensor(randn(Float32, 10, 4, 5))
        y = forward(lstm_seq, x)
        @test size(y.data) == (20, 4, 5)
        
        lstm_last = LSTM(10, 20; return_sequences=false)
        y = forward(lstm_last, x)
        @test size(y.data) == (20, 4)
        
        gru_seq = GRU(10, 20; return_sequences=true)
        y = forward(gru_seq, x)
        @test size(y.data) == (20, 4, 5)
        
        gru_last = GRU(10, 20; return_sequences=false)
        y = forward(gru_last, x)
        @test size(y.data) == (20, 4)
        
        println("  ✅ Shapes correctos")
    end
    
    @testset "2. Cell State Stability" begin
        println("\n2️⃣ Test de estabilidad del cell state")
        
        lstm_cell = LSTMCell(10, 20)
        x = Tensor(randn(Float32, 10, 8))
        h = Tensor(zeros(Float32, 20, 8))
        c = Tensor(zeros(Float32, 20, 8))
        
        for _ in 1:100
            h, c = forward(lstm_cell, x, (h, c))
        end
        
        @test !any(isnan.(c.data))
        @test !any(isinf.(c.data))
        @test maximum(abs.(c.data)) < 100
        
        println("  ✅ Cell state estable después de 100 iteraciones")
    end
    
    @testset "3. Gradient Flow" begin
        println("\n3️⃣ Test de flujo de gradientes")
        
        model = Sequential([
            LSTM(10, 20),
            Dense(20, 5)
        ])
        
        X = Tensor(randn(Float32, 10, 8, 15); requires_grad=true)
        y = Tensor(randn(Float32, 5, 8))
        
        pred = forward(model, X)
        loss = mse_loss(pred, y)
        backward(loss, Tensor(ones(Float32, size(loss.data))))
        
        params = collect_parameters(model)
        for p in params
            @test p.grad !== nothing
            @test !any(isnan.(p.grad.data))
            @test !any(isinf.(p.grad.data))
        end
        
        @test X.grad !== nothing
        println("  ✅ Gradientes fluyen correctamente")
    end
    
    @testset "4. Training Convergence" begin
        println("\n4️⃣ Test de convergencia en entrenamiento")
        
        seq_len = 10
        batch_size = 16
        input_size = 5
        output_size = 3
        
        X = Tensor(randn(Float32, input_size, batch_size, seq_len))
        y = Tensor(randn(Float32, output_size, batch_size))
        
        model = Sequential([
            LSTM(input_size, 32),
            Dense(32, output_size)
        ])
        
        optimizer = Adam(0.01f0)
        losses = Float32[]
        
        for _ in 1:20
            zero_grad!(model)
            pred = forward(model, X)
            loss = mse_loss(pred, y)
            push!(losses, loss.data[1])
            backward(loss, Tensor(ones(Float32, size(loss.data))))
            
            params = collect_parameters(model)
            total_norm = sqrt(sum(sum(abs2, p.grad.data) for p in params if p.grad !== nothing))
            if total_norm > 5f0
                clip_coef = 5f0 / total_norm
                for p in params
                    if p.grad !== nothing
                        p.grad.data .*= clip_coef
                    end
                end
            end
            
            optim_step!(optimizer, params)
        end
        
        @test losses[end] < losses[1]
        @test losses[end] < 0.5 * losses[1]
        println("  ✅ Loss decrece: $(losses[1]) → $(losses[end])")
    end
    
    @testset "5. Bidirectional Processing" begin
        println("\n5️⃣ Test de procesamiento bidireccional")
        
        lstm_forward = LSTM(10, 20; return_sequences=true)
        lstm_backward = LSTM(10, 20; return_sequences=true)
        
        x = Tensor(randn(Float32, 10, 4, 8))
        
        h_forward = forward(lstm_forward, x)
        x_reversed = Tensor(reverse(x.data, dims=3))
        h_backward = forward(lstm_backward, x_reversed)
        h_backward_rev = Tensor(reverse(h_backward.data, dims=3))
        
        h_bidirectional = Tensor(cat(h_forward.data, h_backward_rev.data, dims=1))
        @test size(h_bidirectional.data) == (40, 4, 8)
        
        println("  ✅ Bidireccional funciona: shape $(size(h_bidirectional.data))")
    end
    
    @testset "6. GRU vs LSTM Performance" begin
        println("\n6️⃣ Comparación GRU vs LSTM")
        
        X = Tensor(randn(Float32, 10, 8, 20))
        y = Tensor(randn(Float32, 5, 8))
        
        lstm_model = Sequential([
            LSTM(10, 30),
            Dense(30, 5)
        ])
        
        gru_model = Sequential([
            GRU(10, 30),
            Dense(30, 5)
        ])
        
        lstm_params = sum(length(p.data) for p in collect_parameters(lstm_model))
        gru_params = sum(length(p.data) for p in collect_parameters(gru_model))
        
        @test gru_params < lstm_params
        println("  ✅ LSTM params: $lstm_params, GRU params: $gru_params")
        println("     GRU tiene $(round(100*(1-gru_params/lstm_params), digits=1))% menos parámetros")
    end
    
    @testset "7. GPU Support" begin
        println("\n7️⃣ Test de soporte GPU")
        
        if CUDA.functional()
            lstm = LSTM(10, 20)
            lstm_gpu = model_to_gpu(lstm)
            
            x_gpu = Tensor(CUDA.randn(Float32, 10, 4, 5))
            y_gpu = forward(lstm_gpu, x_gpu)
            
            @test y_gpu.data isa CUDA.CuArray
            @test size(y_gpu.data) == (20, 4)
            
            loss = sum(y_gpu)
            backward(loss, Tensor(CUDA.ones(Float32, 1)))
            
            params = collect_parameters(lstm_gpu)
            for p in params
                @test p.grad !== nothing
                @test p.grad.data isa CUDA.CuArray
            end
            
            println("  ✅ GPU funciona correctamente")
        else
            println("  ⚠️ GPU no disponible, test omitido")
        end
    end
    
    @testset "8. Save/Load Models" begin
        println("\n8️⃣ Test de guardado/carga")
        
        model = Sequential([
            LSTM(5, 10),
            GRU(10, 15),
            Dense(15, 3)
        ])
        
        X = Tensor(randn(Float32, 5, 2, 8))
        y_original = forward(model, X)
        
        save_model(model, "test_lstm_gru.jld2")
        loaded_model = load_model("test_lstm_gru.jld2")
        y_loaded = forward(loaded_model, X)
        
        @test isapprox(y_original.data, y_loaded.data, rtol=1e-5)
        rm("test_lstm_gru.jld2")
        
        println("  ✅ Save/Load funciona correctamente")
    end
    
    @testset "9. Integration with fit!" begin
        println("\n9️⃣ Test de integración con fit!")
        
        n_samples = 50
        X = [Tensor(randn(Float32, 10, 1, 15)) for _ in 1:n_samples]
        y = [Tensor(randn(Float32, 5, 1)) for _ in 1:n_samples]
        
        model = Sequential([
            LSTM(10, 20),
            Dense(20, 5)
        ])
        
        history = fit!(model, X, y,
            epochs=5,
            batch_size=10,
            verbose=false
        )
        
        @test length(history.train_loss) == 5
        @test history.train_loss[end] < history.train_loss[1]
        
        println("  ✅ fit! funciona con LSTM")
    end
    
    @testset "10. Forget Gate Initialization" begin
        println("\n🔟 Test de inicialización del forget gate")
        
        lstm_cell = LSTMCell(10, 20)
        
        @test all(lstm_cell.b_if.data .≈ 1f0)
        @test all(lstm_cell.b_ii.data .≈ 0f0)
        @test all(lstm_cell.b_ig.data .≈ 0f0)
        @test all(lstm_cell.b_io.data .≈ 0f0)
        
        println("  ✅ Forget gate bias = 1.0 (correcto)")
    end
end

println("\n" * "="^50)
println("✅ TODOS LOS TESTS DE LSTM/GRU PASARON")
println("="^50)
