using DeepDynamics
using Test
using DeepDynamics.TensorEngine
@testset "Integración CNN" begin
    # Primero crear las capas convolucionales
    conv_layers = [
        Conv2D(3, 16, (3,3)),
        BatchNorm(16),
        Activation(relu),
        MaxPooling((2,2)),
        Flatten()
    ]
    
    # Calcular la dimensión de salida después de Flatten
    dummy_input = Tensor(randn(Float32, 8, 3, 64, 64))
    x = dummy_input
    for layer in conv_layers
        x = layer(x)
    end
    flat_dim = size(x.data, 1)  # Dimensión después de Flatten
    
    # Ahora crear el modelo completo con la dimensión correcta
    model = Sequential([
        conv_layers...,
        Dense(flat_dim, 10),  # Usar la dimensión calculada
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
