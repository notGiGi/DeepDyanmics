using DeepDynamics
using Random
Random.seed!(123)

println("=== Test de Entrenamiento Completo ===")

n = 100
X_data = randn(Float32, 1, n)
y_data = 2.0f0 .* X_data .+ 1.0f0 .+ 0.1f0 .* randn(Float32, 1, n)

X = Tensor(X_data; requires_grad=false)
y = Tensor(y_data; requires_grad=false)

model = Sequential([Dense(1, 1)])
opt = SGD(learning_rate=0.01)

losses = Float32[]
println("\nEntrenando...")
for epoch in 1:50
    params = collect_parameters(model)
        for p in params
        zero_grad!(p)
    end
        pred = model(X)
    loss = mse_loss(pred, y)
    push!(losses, loss.data[1])
    backward(loss, Tensor([1.0f0]))
        DeepDynamics.optim_step!(opt, params)
    if epoch % 10 == 0
        println("  Epoch $epoch: loss = ", loss.data[1])
            end
end

println("\nResultados:")
println("  Pérdida inicial: ", losses[1])
println("  Pérdida final: ", losses[end])
println("  Reducción: ", round((1 - losses[end]/losses[1])*100, digits=2), "%")

w = Array(model.layers[1].weights.data)[1]  # Traer a CPU primero
b = Array(model.layers[1].biases.data)[1]   # Traer a CPU primero
println("\nParámetros aprendidos:")
println("  Peso: $w (esperado ≈ 2.0)")
println("  Bias: $b (esperado ≈ 1.0)")

@assert losses[end] < losses[1] "La pérdida debe disminuir"
println("\n✓ Entrenamiento funciona correctamente")
