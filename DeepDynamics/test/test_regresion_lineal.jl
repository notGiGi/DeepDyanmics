using DeepDynamics
using Test

@testset "Regresion Lineal" begin
    X = Tensor(randn(Float32, 1, 100))
    y_true = 3 .* X .+ 5 .+ randn(Float32, 1, 100) * 0.1

    model = Sequential([Dense(1, 1)])
    y_pred = model(X)
    loss = mse_loss(y_pred, y_true)

    params = collect_parameters(model)
    for p in params
        zero_grad!(p)
    end

    backward(loss, [1.0f0])
    for p in params
        @test p.grad !== nothing
        @test !any(isnan.(p.grad.data))
    end
end
