using DeepDynamics
using Random
Random.seed!(123)

println("=== TEST INTEGRACIÓN FASE 1 + FASE 2 ===")

# Datos: problema de regresión simple
n_samples = 100
X_data = randn(Float32, 5, n_samples)
y_data = sum(X_data, dims=1) .+ 0.1f0 .* randn(Float32, 1, n_samples)

X = Tensor(X_data; requires_grad=false)
y = Tensor(y_data; requires_grad=false)

# Modelo más complejo
model = Sequential([
    Dense(5, 10),
    Activation(relu),
        Dense(10, 5),
    Activation(relu),
        Dense(5, 1)
])

# Adam mejorado
opt = Adam(learning_rate=0.01, weight_decay=0.001)
params = collect_parameters(model)

# Entrenar
losses = Float32[]
for epoch in 1:100
    # zero_grad!
        for p in params
                zero_grad!(p)
                    end

    # Forward
        pred = model(X)
    loss = mse_loss(pred, y)
    push!(losses, loss.data[1])
    
        # Backward
            backward(loss, [1.0f0])
            
                # Step
    DeepDynamics.optim_step!(opt, params)

    if epoch % 20 == 0
        println("  Epoch $epoch: loss = ", round(loss.data[1], digits=4))
            end
end

# Verificar mejora
improvement = (losses[1] - losses[end]) / losses[1] * 100
println("\nMejora total: $(round(improvement, digits=1))%")
@assert losses[end] < losses[1] * 0.5 "No hubo suficiente mejora"

# Test robustez: NaN
println("\nTest: Robustez ante NaN")
old_param = copy(params[1].data)
params[1].grad = Tensor(fill(NaN32, size(params[1].data)...); requires_grad=false)
DeepDynamics.optim_step!(opt, params)
@assert params[1].data == old_param "Parámetros cambiaron con NaN"

println("\n✓ INTEGRACIÓN FASE 1 + FASE 2 EXITOSA")
