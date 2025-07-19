
using DeepDynamics
using DeepDynamics.TensorEngine: Tensor, add, backward, mse_loss, zero_grad!
using DeepDynamics.NeuralNetwork: Sequential, Dense, forward, collect_parameters
using DeepDynamics.Optimizers: SGD, step!

println("=== TEST FASE 1: zero_grad! y requires_grad ===")

# Test 1: Verificar que zero_grad! funciona
println("\nTest 1: zero_grad!")
x = Tensor(randn(Float32, 5, 5))
x.grad = Tensor(ones(Float32, 5, 5); requires_grad=false)
println("Grad antes: sum = ", sum(x.grad.data))
zero_grad!(x)
println("Grad después: sum = ", sum(x.grad.data))
@assert sum(x.grad.data) == 0.0f0 "zero_grad! falló"
println("✓ zero_grad! funciona correctamente")

# Test 2: Verificar requires_grad
println("\nTest 2: requires_grad")
x_with_grad = Tensor(randn(Float32, 3, 3); requires_grad=true)
x_no_grad = Tensor(randn(Float32, 3, 3); requires_grad=false)
println("Tensor con grad: requires_grad = ", x_with_grad.requires_grad)
println("Tensor sin grad: requires_grad = ", x_no_grad.requires_grad)
println("✓ requires_grad se inicializa correctamente")

# Test 3: Verificar que backward respeta requires_grad
println("\nTest 3: backward respeta requires_grad")
a = Tensor([1.0f0, 2.0f0]; requires_grad=true)
b = Tensor([3.0f0, 4.0f0]; requires_grad=false)
c = add(a, b)

# Backward con gradiente apropiado
backward(c, Tensor([1.0f0, 1.0f0]))

println("a.grad existe: ", a.grad !== nothing)
println("b.grad existe: ", b.grad !== nothing)
@assert a.grad !== nothing "a debería tener gradiente"
@assert b.grad === nothing "b NO debería tener gradiente"
println("✓ backward respeta requires_grad")

# Test 4: Verificar que el entrenamiento funciona
println("\nTest 4: Entrenamiento básico")
# Modelo simple
model = Sequential([
    Dense(2, 1)
])

# Datos simples
X = Tensor(randn(Float32, 2, 10); requires_grad=false)
y = Tensor(randn(Float32, 1, 10); requires_grad=false)

# Optimizer
opt = SGD(learning_rate=0.01)
params = collect_parameters(model)

# Guardar pesos iniciales
initial_weights = copy(params[1].data)

# Entrenar por 5 iteraciones
losses = Float32[]
for i in 1:5
    # CRÍTICO: zero_grad antes de forward
    for p in params
        zero_grad!(p)
    end
    
    # Forward
    pred = forward(model, X)
    loss = mse_loss(pred, y)
    push!(losses, loss.data[1])
    
    # Backward - gradiente escalar
    backward(loss, [1.0f0])
    
    # Update
    step!(opt, params)
end

println("Losses: ", losses)
println("Pesos iniciales vs finales diferentes: ", !isapprox(initial_weights, params[1].data))

# Verificar que algo cambió
@assert !isapprox(initial_weights, params[1].data) "Los pesos no cambiaron!"
println("✓ Entrenamiento básico funciona")

# Test 5: Test de propagación escalar
println("\nTest 5: Propagación de gradiente escalar")
x = Tensor([2.0f0, 3.0f0]; requires_grad=true)
y = Tensor([1.0f0, 1.0f0]; requires_grad=false)
loss = mse_loss(x, y)

println("Loss shape: ", size(loss.data))
println("Loss value: ", loss.data[1])

zero_grad!(x)
backward(loss, [1.0f0])

println("x.grad: ", x.grad.data)
expected_grad = 2.0f0 * (x.data .- y.data) / length(x.data)
println("Expected grad: ", expected_grad)
@assert isapprox(x.grad.data, expected_grad, rtol=1e-5) "Gradiente incorrecto"
println("✓ Propagación escalar funciona")

println("\n=== TODOS LOS TESTS DE FASE 1 PASARON ===")
