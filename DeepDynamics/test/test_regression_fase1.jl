using DeepDynamics
using Test

@testset "Regresión Básica FASE 1" begin
    x = Tensor(randn(Float32, 10, 10))
    y = Tensor(randn(Float32, 10, 10))
    z = add(x, y)
    @test size(z.data) == (10, 10)
    
    layer = Dense(10, 5)
    input = Tensor(randn(Float32, 10, 2))
    output = layer(input)
        @test size(output.data) == (5, 2)
    
    loss = mse_loss(output, Tensor(randn(Float32, 5, 2)))
    zero_grad!(layer.weights)
    backward(loss, Tensor([1.0f0]))
    @test layer.weights.grad !== nothing
    
    opt = SGD(learning_rate=0.01)
    old_weights = copy(layer.weights.data)
        DeepDynamics.optim_step!(opt, [layer.weights])
    @test !isapprox(old_weights, layer.weights.data)
    end

println("✓ Tests de regresión FASE 1 pasaron!")
