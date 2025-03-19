using Test
using DeepDynamics
using Random
using Statistics

# Importamos explícitamente los símbolos necesarios desde los submódulos:
import DeepDynamics.TensorEngine: Tensor, add, matmul, mse_loss, initialize_weights, l2_regularization, clip_gradients!
import DeepDynamics: forward, collect_parameters, relu, sigmoid, tanh_activation, leaky_relu
import DeepDynamics.NeuralNetwork: swish, mish
import DeepDynamics.Utils: normalize_inputs
import DeepDynamics.Optimizers: step!
import DeepDynamics: binary_crossentropy, build_vocabulary, text_to_indices, pad_sequence, embedding_forward, conv_forward
import DeepDynamics.Conv2D, DeepDynamics.MaxPooling

Random.seed!(1234)  # Para reproducibilidad

#####################################################################
# Tests para el módulo TensorEngine y funciones básicas
#####################################################################
@testset "TensorEngine Tests" begin
    @testset "Tensor Addition" begin
        a = Tensor(reshape([1.0, 2.0, 3.0], (3,1)))
        b = Tensor(reshape([4.0, 5.0, 6.0], (3,1)))
        result = add(a, b)
        @test result.data == reshape([5.0, 7.0, 9.0], (3,1))
    end

    @testset "Matrix Multiplication" begin
        a = Tensor([1.0 2.0; 3.0 4.0])
        b = Tensor([5.0 6.0; 7.0 8.0])
        result = matmul(a, b)
        @test result.data == [19.0 22.0; 43.0 50.0]
    end

    @testset "MSE Loss" begin
        y_pred = Tensor(reshape([2.0, 3.0], (2,1)))
        y_true = Tensor(reshape([1.0, 5.0], (2,1)))
        loss = mse_loss(y_pred, y_true)
        expected = sum(([2.0, 3.0] .- [1.0, 5.0]).^2) / 2
        @test loss.data[1] ≈ expected atol=1e-6
    end

    @testset "Initialize Weights" begin
        weights = initialize_weights((3, 4); method=:xavier)
        @test size(weights.data) == (4, 3)  # (fan_out, fan_in)
    end

    @testset "L2 Regularization" begin
        dummy_weight = Tensor(reshape([1.0, 2.0], (2,1)))
        reg = l2_regularization([dummy_weight], 0.1)
        expected = 0.1 * (1.0^2 + 2.0^2)
        @test reg.data[1] ≈ expected atol=1e-6
    end

    @testset "Normalize Inputs" begin
        inputs = [Tensor(randn(5,1)) for _ in 1:3]
        norm_inputs = normalize_inputs(inputs)
        for input in norm_inputs
            m = Statistics.mean(input.data)
            s = Statistics.std(input.data)
            @test m ≈ 0 atol=1e-6
            @test s ≈ 1 atol=1e-6
        end
    end

    @testset "Gradient Clipping" begin
        t = Tensor(randn(10,10))
        t.grad = Tensor(randn(10,10) .* 100)
        clip_gradients!(t, 50.0)
        @test all(abs.(t.grad.data) .<= 50.0)
    end

    @testset "Optimizer Step (Dummy)" begin
        opt = DeepDynamics.Optimizers.SGD(learning_rate=0.1)
        p = Tensor(reshape([1.0, 2.0], (2,1)))
        p.grad = Tensor(reshape([0.5, 0.5], (2,1)))
        step!(opt, [p])
        @test p.data ≈ reshape([0.95, 1.95], (2,1)) atol=1e-6
    end
end

#####################################################################
# Tests para el módulo NeuralNetwork y Layers
#####################################################################
@testset "NeuralNetwork & Layers Tests" begin
    @testset "Dense Layer Forward" begin
        layer = DeepDynamics.Dense(2, 2, activation=relu)
        input = Tensor(reshape([1.0, 2.0], (2,1)))
        model = DeepDynamics.Sequential([layer])
        output = forward(model, input; verbose=false)
        @test size(output.data) == (2,1)
    end

    @testset "Sequential Forward (Funciones simples)" begin
        model = DeepDynamics.Sequential([
            (input) -> relu(input),
            (input) -> sigmoid(input)
        ])
        input = Tensor(reshape([-1.0, 0.0, 1.0], (3,1)))
        output = forward(model, input; verbose=false)
        @test size(output.data) == (3,1)
    end

    @testset "Activation Functions" begin
        t = Tensor(reshape([-1.0, 0.0, 1.0], (3,1)))
        r = relu(t)
        @test all(r.data .>= 0)
        s = sigmoid(t)
        @test all(s.data .> 0 .&& s.data .< 1)
        tanh_out = tanh_activation(t)
        @test all(abs.(tanh_out.data) .<= 1)
        lr = leaky_relu(t; α=0.01)
        @test any(lr.data .< 0)
        sw = swish(t)
        @test size(sw.data) == size(t.data)
        mi = mish(t)
        @test size(mi.data) == size(t.data)
    end

    @testset "Flatten Layer" begin
        x = Tensor(randn(4,3,2))
        flat = DeepDynamics.Layers.forward(DeepDynamics.Flatten(), x)
        @test prod(size(x.data)) == length(flat.data)
    end
end

#####################################################################
# Tests para ConvolutionalLayers y EmbeddingLayer
#####################################################################
@testset "Convolution & Embedding Tests" begin
    @testset "Embedding Forward" begin
        emb = DeepDynamics.Embedding(100, 16)
        indices = [1, 2, 0, 5]
        out = embedding_forward(emb, indices)
        @test size(out.data) == (16, length(indices))
    end

    @testset "Conv2D Forward" begin
        dummy_input = Tensor(randn(10,10,1))
        conv = DeepDynamics.Conv2D(1, 2, (3,3); stride=1, padding=(0,0))
        out_conv = conv_forward(conv, dummy_input)
        @test size(out_conv.data) == (8,8,2)
    end

    @testset "MaxPooling Forward" begin
        dummy_conv = Tensor(randn(8,8,2))
        pool = DeepDynamics.MaxPooling((2,2); stride=(2,2))
        # Se invoca conv_forward ya que la implementación de MaxPooling está en ConvolutionalLayers
        out_pool = conv_forward(pool, dummy_conv)
        @test size(out_pool.data) == (4,4,2)
    end
end

#####################################################################
# Tests para Optimizers
#####################################################################
@testset "Optimizers Tests" begin
    x = Tensor(reshape([1.0,2.0], (2,1)))
    x.grad = Tensor(reshape([0.1,0.1], (2,1)))
    
    sgd_opt = DeepDynamics.Optimizers.SGD(learning_rate=0.1)
    sgd_x = deepcopy(x)
    step!(sgd_opt, [sgd_x])
    @test sgd_x.data ≈ reshape([0.99,1.99], (2,1)) atol=1e-6

    adam_opt = DeepDynamics.Optimizers.Adam(learning_rate=0.01)
    adam_x = deepcopy(x)
    adam_x.grad = Tensor(reshape([0.1,0.1], (2,1)))
    step!(adam_opt, [adam_x])
    @test size(adam_x.data) == size(x.data)
    
    rms_opt = DeepDynamics.Optimizers.RMSProp(learning_rate=0.01)
    rms_x = deepcopy(x)
    rms_x.grad = Tensor(reshape([0.1,0.1], (2,1)))
    step!(rms_opt, [rms_x])
    @test size(rms_x.data) == size(x.data)
    
    adagrad_opt = DeepDynamics.Optimizers.Adagrad(learning_rate=0.01)
    ada_x = deepcopy(x)
    ada_x.grad = Tensor(reshape([0.1,0.1], (2,1)))
    step!(adagrad_opt, [ada_x])
    @test size(ada_x.data) == size(x.data)
    
    nadam_opt = DeepDynamics.Optimizers.Nadam(learning_rate=0.01)
    nadam_x = deepcopy(x)
    nadam_x.grad = Tensor(reshape([0.1,0.1], (2,1)))
    step!(nadam_opt, [nadam_x])
    @test size(nadam_x.data) == size(x.data)
end

#####################################################################
# Tests para Training y Metrics
#####################################################################
@testset "Training & Metrics Tests" begin
    y_pred = Tensor(reshape([0.8], (1,1)))
    y_true = Tensor(reshape([1.0], (1,1)))
    loss = binary_crossentropy(y_pred, y_true)
    @test loss.data[1] > 0

    dummy_model = DeepDynamics.Sequential([(input) -> input])
    dummy_input = Tensor(reshape([0.7], (1,1)))
    dummy_target = Tensor(reshape([1.0], (1,1)))
    acc = DeepDynamics.Training.compute_accuracy(dummy_model, [dummy_input], [dummy_target])
    @test acc == 1.0
end

#####################################################################
# Tests para TextUtils
#####################################################################
@testset "TextUtils Tests" begin
    texts = ["This is a test", "Another test test"]
    vocab = build_vocabulary(texts, 100)
    @test typeof(vocab) == Dict{String,Int}
    indices = text_to_indices("This is a test", vocab)
    @test isa(indices, Vector{Int})
    padded = pad_sequence(indices, 10)
    @test length(padded) == 10
end

#####################################################################
# Tests para Layers y Utils
#####################################################################
@testset "Layers & Utils Tests" begin
    dropout_layer = DeepDynamics.Layers.Dropout(0.5; training=true)
    x = Tensor(randn(4,4))
    out_dropout = DeepDynamics.Layers.forward(dropout_layer, x)
    @test size(out_dropout.data) == size(x.data)
    
    bn = DeepDynamics.Layers.BatchNorm(4; training=true)
    bn_out = DeepDynamics.Layers.forward(bn, x)
    @test size(bn_out.data) == size(x.data)
    
    flat = DeepDynamics.Layers.forward(DeepDynamics.Flatten(), x)
    @test prod(size(x.data)) == length(flat.data)
    
    gap = DeepDynamics.Layers.global_average_pooling(x)
    @test size(gap.data,2) == 1
end

println("Todos los tests se ejecutaron (se han ampliado para cubrir la mayor parte de la paquetería).")
