using DeepDynamics
using Test

@testset "Integracin CNN" begin
    model = Sequential([
        Conv2D(3, 16, (3,3)),
        BatchNorm(16),
        Activation(relu),
        MaxPooling((2,2)),
        Flatten(),
        Dense(16*31*31, 10),
        Activation(softmax)
    ])

    X = Tensor(randn(8, 3, 64, 64))
    y = Tensor(rand(10, 8))
    output = model(X)
    @test size(output.data) == (10, 8)

    loss = categorical_crossentropy(output, y)
    params = collect_parameters(model)

    for p in params
        if isdefined(TensorEngine, :zero_grad!)
            zero_grad!(p)
        end
    end

    backward(loss, Tensor([1.0]))

    for p in params
        if p.requires_grad
            @test p.grad !== nothing
            @test !any(isnan.(p.grad.data))
        end
    end
end
