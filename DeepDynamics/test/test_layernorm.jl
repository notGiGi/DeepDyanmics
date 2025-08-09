# test/test_layernorm_production.jl
using Test
using DeepDynamics
using Statistics

@testset "LayerNorm Production Tests" begin
    # Test 1: Formato 2D (features, batch)
    @testset "2D Format" begin
        x = Tensor(randn(Float32, 20, 32))  # 20 features, 32 samples
        ln = LayerNorm(20)
        y = forward(ln, x)
        
        # Verificar normalización
        for i in 1:32
            sample = y.data[:, i]
            @test abs(mean(sample)) < 1e-5
            @test abs(std(sample) - 1.0) < 0.1
        end
    end
    
    # Test 2: Formato 4D NCHW
    @testset "4D NCHW Format" begin
        x = Tensor(randn(Float32, 4, 16, 8, 8))  # (batch, channels, H, W)
        ln = LayerNorm(16)  # Normalizar sobre channels
        y = forward(ln, x)
        
        @test size(y.data) == size(x.data)
        @test !any(isnan.(y.data))
    end
    
    # Test 3: Integración completa
    @testset "Full Integration" begin
        model = Sequential([
            Dense(10, 20),
            LayerNorm(20),
            Activation(relu),
            Dense(20, 5)
        ])
        
        X = Tensor(randn(Float32, 10, 32))
        y = Tensor(randn(Float32, 5, 32))
        
        # Training
        optimizer = Adam(0.01f0)
        losses = Float32[]
        
        for epoch in 1:10
            zero_grad!(model)
            pred = forward(model, X)
            loss = mse_loss(pred, y)
            push!(losses, loss.data[1])
            backward(loss, Tensor(ones(Float32, size(loss.data))))
            params = collect_parameters(model)
            optim_step!(optimizer, params)
        end
        
        @test losses[end] < losses[1]
    end
end

println("✅ LayerNorm implementación de producción completa")