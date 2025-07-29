using DeepDynamics
using Test
using DeepDynamics.TensorEngine
using DeepDynamics: Activation

@testset "Integración CNN" begin
    conv_layers = [
        Conv2D(3, 16, (3,3)),
        BatchNorm(16),
        Activation(relu),
        MaxPooling((2,2)),
        Flatten()
    ]

    dummy_input = Tensor(randn(Float32, 8, 3, 64, 64))
    x = dummy_input
    for layer in conv_layers
        x = layer(x)
    end
    flat_dim = size(x.data, 1)

    model = Sequential([
        conv_layers...,
        Dense(flat_dim, 10),
        Activation(softmax)
    ])

    X = Tensor(randn(Float32, 8, 3, 64, 64))
    y = Tensor(rand(Float32, 10, 8))

    output = model(X)
    @test size(output.data) == (10, 8)

    loss = categorical_crossentropy(output, y)
    params = collect_parameters(model)

    for p in params
        zero_grad!(p)
    end

    backward(loss, Tensor([1.0f0]))

    for p in params
        if p.requires_grad
            @test p.grad !== nothing
            @test !any(isnan.(p.grad.data))
        end
    end
end
