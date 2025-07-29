using DeepDynamics
using DeepDynamics.TensorEngine
using DeepDynamics.Optimizers    # para Adam
using Printf
using NNlib: σ
import DeepDynamics: binary_crossentropy_with_logits

# ————————————————————————————————————————————————————————
# 1) Generar datos de espiral igual que en el integration test
n = 200
X_data = Float32[]
y_data = Float32[]
for i in 1:n
    angle  = 4π * i / n
    radius = i / n
    if iseven(i)
        x1, x2 =  radius*cos(angle),  radius*sin(angle)
        push!(y_data, 1f0)
    else
        x1, x2 = -radius*cos(angle), -radius*sin(angle)
        push!(y_data, 0f0)
    end
    append!(X_data, (x1 + 0.1f0*randn(Float32),
                    x2 + 0.1f0*randn(Float32)))
end
X_list = [ Tensor(X_data[2i-1:2i]) for i in 1:n ]
y_list = [ Tensor([y_data[i]])      for i in 1:n ]

# 2) Modelo SIN la última sigmoid: entregará “logits”
model = Sequential([
    Dense(2, 32; init_method=:he),
    Activation(relu),
    Dense(32,32; init_method=:he),
    Activation(relu),
    Dense(32,16; init_method=:he),
    Activation(relu),
    Dense(16,1;  init_method=:he)
    # <-- NO sigmoid aquí
])

# 3) Entrenamiento manual usando Adam + logits‐BCE
opt = Adam(learning_rate=0.05f0)
params = collect_parameters(model)

println("Epoch |    Loss    |   Accuracy   ")
println("-----------------------------------")

for epoch in 1:200
    # zero grads
    for p in params
        zero_grad!(p)
    end

    # forward
    logits = forward(model, stack_batch(X_list))
    # loss with logits
    loss   = binary_crossentropy_with_logits(logits, stack_batch(y_list))
    # backward
    TensorEngine.backward(loss, ones(size(loss.data)))
    # update
    step!(opt, params)

    # compute accuracy on batch
    preds = σ.(logits.data) .> 0.5f0
    acc   = sum(preds .== (stack_batch(y_list).data .> 0.5f0)) / n

    if epoch % 50 == 0
        @printf("%5d | %10.4f | %10.3f%%\n",
                epoch, loss.data[1], acc*100)
    end
end
