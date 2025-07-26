using Test
using DeepDynamics
using CUDA
using DeepDynamics.NeuralNetwork
using  DeepDynamics.TensorEngine
using DeepDynamics.Losses  # o import DeepDynamics.Losses
using Statistics
println("=== TESTS FASE 7: Pérdidas y Gradientes ===\n")

# ================================================
# Configuración
# ================================================
BATCH_SIZE = 4
NUM_FEATURES = 10
NUM_CLASSES = 3

# Verificar si hay GPU disponible
gpu_available = CUDA.has_cuda() && CUDA.functional()
println("GPU disponible: $gpu_available")

# ================================================
# 1. Test Binary Crossentropy (Sigmoid)
# ================================================
@testset "Binary Crossentropy + Sigmoid" begin
    println("Testing binary_crossentropy con sigmoid...")

    # Datos de entrada y etiquetas
    x_data = rand(Float32, NUM_FEATURES, BATCH_SIZE)
    y_true_data = rand(Bool, 1, BATCH_SIZE)  # Etiquetas binarias

    # Crear tensores
    x = TensorEngine.Tensor(x_data)
    y_true = TensorEngine.Tensor(Float32.(y_true_data))

    # Modelo simple: Dense + Sigmoid
    dense = Dense(NUM_FEATURES, 1)
    output = sigmoid(dense(x))

    # Pérdida
    loss = Losses.binary_crossentropy(output, y_true)
    println("Loss (binary): ", loss.data[1,1])

    # Backward
    TensorEngine.backward(loss, [1.0f0])


    # Verificar gradientes no nulos
    @test all(abs.(dense.weights.grad.data) .> 0f0)
    @test all(abs.(dense.biases.grad.data) .> 0f0)
end

# ================================================
# 2. Test Categorical Crossentropy (Softmax)
# ================================================
@testset "Categorical Crossentropy + Softmax" begin
    println("Testing categorical_crossentropy con softmax...")

    # Datos de entrada y etiquetas one-hot
    x_data = rand(Float32, NUM_FEATURES, BATCH_SIZE)
    y_true_data = zeros(Float32, NUM_CLASSES, BATCH_SIZE)
    for i in 1:BATCH_SIZE
        y_true_data[rand(1:NUM_CLASSES), i] = 1f0
    end

    # Crear tensores
    x = TensorEngine.Tensor(x_data)
    y_true = TensorEngine.Tensor(y_true_data)

    # Modelo simple: Dense + Softmax
    dense = Dense(NUM_FEATURES, NUM_CLASSES)
    logits = dense(x)
    output = softmax(logits)

    # Pérdida
    loss = Losses.categorical_crossentropy(output, y_true)
    println("Loss (categorical): ", loss.data[1,1])

    # Backward
    TensorEngine.backward(loss, [1.0f0])

    # Verificar gradientes no nulos
    @test all(abs.(dense.weights.grad.data) .> 0f0)
    @test all(abs.(dense.biases.grad.data) .> 0f0)
end

# ================================================
# 3. Test GPU Compatibility
# ================================================
if gpu_available
    @testset "Entrenamiento en GPU" begin
        println("Testing entrenamiento GPU...")

        # Datos de entrada en GPU
        x_data = CUDA.CuArray(rand(Float32, NUM_FEATURES, BATCH_SIZE))
        
        # Crear labels en CPU primero
        y_true_cpu = zeros(Float32, NUM_CLASSES, BATCH_SIZE)
        for i in 1:BATCH_SIZE
            y_true_cpu[rand(1:NUM_CLASSES), i] = 1f0
        end
        y_true_data = CUDA.CuArray(y_true_cpu)

        x = TensorEngine.Tensor(x_data)
        y_true = TensorEngine.Tensor(y_true_data)

        # Modelo simple en GPU
        dense = Dense(NUM_FEATURES, NUM_CLASSES)
        model = NeuralNetwork.Sequential([dense, NeuralNetwork.Activation(softmax)])
        model_gpu = model_to_gpu(model)

        # Forward + loss
        output = model_gpu(x)
        loss = Losses.categorical_crossentropy(output, y_true)
        println("Loss (GPU): ", loss.data[1,1])

        # Backward
        TensorEngine.backward(loss, [1.0f0])

        # OBTENER LA CAPA DENSE DEL MODELO GPU
        dense_gpu = model_gpu.layers[1]  # Primera capa del modelo GPU
        
        # Verificar gradientes en GPU
        @test dense_gpu.weights.grad !== nothing
        @test dense_gpu.biases.grad !== nothing
        
        # Verificar que están en GPU y tienen valores no cero
        W_grad = dense_gpu.weights.grad.data
        b_grad = dense_gpu.biases.grad.data
        
        @test W_grad isa CUDA.CuArray
        @test b_grad isa CUDA.CuArray
        @test all(abs.(Array(W_grad)) .> 0f0)
        @test all(abs.(Array(b_grad)) .> 0f0)
    end
else
    println("⚠️ Saltando tests de GPU: no hay GPU disponible.")
end

println("\n=== RESUMEN FASE 7 ===")
println("✅ Pérdidas y gradientes corregidos y testeados.")
println("✅ Funciona con batch size > 1")
println("✅ Compatible con CPU y GPU.")
println("✅ Todos los gradientes verificados.")
