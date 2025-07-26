# test_debug_gradients.jl
using DeepDynamics
using CUDA

# Crear un modelo simple (como en integration_test.jl)
model = Sequential([
    Conv2D(3, 16, (3,3)),
    Activation(relu),
    Flatten(),
    Dense(16*30*30, 10)
])

# Mover a GPU si está disponible
if CUDA.functional()
    model = DeepDynamics.NeuralNetwork.model_to_gpu(model)
end

# Datos de prueba
x = Tensor(randn(Float32, 1, 3, 32, 32); requires_grad=true)
y = Tensor(randn(Float32, 10, 1); requires_grad=false)  # Target para MSE

# Forward
println("=== Forward pass ===")
out = model(x)
println("Output shape: $(size(out.data))")
println("Output requires_grad: $(out.requires_grad)")

# Loss - usar MSE como en los tests
loss = mse_loss(out, y)
println("\n=== Loss ===")
println("Loss value: $(loss.data[1])")
println("Loss requires_grad: $(loss.requires_grad)")
println("Loss has backward_fn: $(loss.backward_fn !== nothing)")

# Backward
if loss.backward_fn !== nothing
    loss.backward_fn([1.0f0])
else
    println("WARNING: Loss no tiene backward_fn!")
end

# Verificar gradientes
println("\n=== Gradientes ===")
params = collect_parameters(model)
for (i, layer) in enumerate(model.layers)
    if layer isa Conv2D
        println("Conv2D: weights.grad=$(layer.weights.grad !== nothing), bias.grad=$(layer.bias.grad !== nothing)")
    elseif layer isa Dense
        println("Dense: weights.grad=$(layer.weights.grad !== nothing), biases.grad=$(layer.biases.grad !== nothing)")
    end
end