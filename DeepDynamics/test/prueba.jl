using DeepDynamics
using DeepDynamics.TensorEngine
using DeepDynamics.NeuralNetwork
using DeepDynamics.Optimizers
using DeepDynamics.Losses
using CUDA

println("=== Test GPU: Red Neuronal Profunda ===")

HAS_GPU = CUDA.has_cuda() && CUDA.functional()
@assert HAS_GPU "❌ No hay GPU disponible."

# =====================================================
# 1. Generación de datos (problema no lineal)
# =====================================================
input_size = 20
hidden_sizes = [64, 32, 16]
output_size = 4
samples = 200
epochs = 25

# Features aleatorios + etiquetas en 4 clases
X_cpu = randn(Float32, input_size, samples)
y_data = zeros(Float32, output_size, samples)
for i in 1:samples
    y_data[rand(1:output_size), i] = 1f0
end

# Enviar datos a GPU
X = Tensor(CUDA.CuArray(X_cpu))
y = Tensor(CUDA.CuArray(y_data))

# =====================================================
# 2. Modelo profundo
# =====================================================
model = model_to_gpu(Sequential([
    Dense(input_size, hidden_sizes[1]),
    LayerActivation(relu),
    Dense(hidden_sizes[1], hidden_sizes[2]),
    LayerActivation(relu),
    Dense(hidden_sizes[2], hidden_sizes[3]),
    LayerActivation(relu),
    Dense(hidden_sizes[3], output_size),
    Activation(softmax)
]))

# =====================================================
# 3. Entrenamiento
# =====================================================
opt = Adam(learning_rate = 0.01)
params = collect_parameters(model)

losses = Float32[]
for epoch in 1:epochs
    for p in params
        zero_grad!(p)
    end

    output = model(X)
    loss = categorical_crossentropy(output, y)
    push!(losses, loss.data[1])
    backward(loss, CUDA.ones(Float32, 1))
    step!(opt, params)

    println("📉 Epoch $epoch - Loss: $(round(loss.data[1]; digits=5))")
end

# =====================================================
# 4. Validación
# =====================================================
@assert losses[end] < losses[1] "❌ El modelo no logró aprender."
println("✅ GPU Deep Model: aprendizaje exitoso, la pérdida final es menor.")
