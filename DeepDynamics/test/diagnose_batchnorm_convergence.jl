# diagnose_batchnorm_convergence.jl
# Script para identificar por qué el modelo con BatchNorm no converge

using DeepDynamics
using Statistics
using DeepDynamics.TensorEngine
using LinearAlgebra
println("="^60)
println("DIAGNÓSTICO: ¿Por qué BatchNorm no converge?")
println("="^60)

# Test 1: Verificar propagación de gradientes
println("\n1️⃣ Test de propagación de gradientes a través de BatchNorm")
x = Tensor(randn(Float32, 4, 2, 8, 8); requires_grad=true)
bn = BatchNorm(2, training=true)

# Forward
y = bn(x)

# Loss simple
loss = sum(y.data)
loss_tensor = Tensor([loss]; requires_grad=true)
loss_tensor.backward_fn = _ -> begin
    TensorEngine.backward(y, ones(size(y.data)))
end

# Zero grad
zero_grad!(x)
zero_grad!(bn.gamma)
zero_grad!(bn.beta)

# Backward
backward(loss_tensor, [1.0f0])

println("¿x tiene gradiente? ", x.grad !== nothing)
println("¿gamma tiene gradiente? ", bn.gamma.grad !== nothing)
println("¿beta tiene gradiente? ", bn.beta.grad !== nothing)

if x.grad !== nothing
    println("Norma del gradiente de x: ", norm(x.grad.data))
    println("¿Gradiente de x contiene NaN? ", any(isnan.(x.grad.data)))
end

# Test 2: Modelo simple con y sin BatchNorm
println("\n2️⃣ Comparación: Con BatchNorm vs Sin BatchNorm")

# Datos simples linealmente separables
X_data = Float32[1 2 3 4; 2 4 6 8]
y_data = Float32[0 0 1 1; 1 1 0 0]
X = Tensor(X_data)
y = Tensor(y_data)

# Modelo SIN BatchNorm
println("\nModelo SIN BatchNorm:")
model_no_bn = Sequential([
    Dense(2, 4),
    Activation(relu),
    Dense(4, 2),
    Activation(softmax)
])

opt1 = Adam(learning_rate=0.1f0)
params1 = collect_parameters(model_no_bn)

losses_no_bn = Float32[]
for i in 1:20
    for p in params1
        zero_grad!(p)
    end
    
    pred = model_no_bn(X)
    loss = categorical_crossentropy(pred, y)
    push!(losses_no_bn, loss.data[1])
    
    backward(loss, [1.0f0])
    optim_step!(opt1, params1)
end

println("  Loss inicial: ", losses_no_bn[1])
println("  Loss final: ", losses_no_bn[end])
println("  Reducción: ", round((1 - losses_no_bn[end]/losses_no_bn[1])*100, digits=1), "%")

# Modelo CON BatchNorm
println("\nModelo CON BatchNorm:")
model_with_bn = Sequential([
    Dense(2, 4),
    BatchNorm(4),
    Activation(relu),
    Dense(4, 2),
    Activation(softmax)
])

opt2 = Adam(learning_rate=0.1f0)
params2 = collect_parameters(model_with_bn)

losses_with_bn = Float32[]
for i in 1:20
    for p in params2
        zero_grad!(p)
    end
    
    pred = model_with_bn(X)
    loss = categorical_crossentropy(pred, y)
    push!(losses_with_bn, loss.data[1])
    
    backward(loss, [1.0f0])
    
    # Verificar gradientes en iteración 1
    if i == 1
        println("\n  Gradientes en primera iteración:")
        for (j, p) in enumerate(params2)
            if p.grad !== nothing
                grad_norm = norm(p.grad.data)
                has_nan = any(isnan.(p.grad.data))
                println("    Param $j: norm=$(grad_norm), NaN=$(has_nan)")
            end
        end
    end
    
    optim_step!(opt2, params2)
end

println("\n  Loss inicial: ", losses_with_bn[1])
println("  Loss final: ", losses_with_bn[end])
println("  Reducción: ", round((1 - losses_with_bn[end]/losses_with_bn[1])*100, digits=1), "%")

# Test 3: Verificar valores de activación
println("\n3️⃣ Análisis de activaciones con BatchNorm")
x_test = Tensor(randn(Float32, 4, 16))
model_debug = Sequential([
    Dense(4, 8),
    BatchNorm(8),
    Activation(relu)
])

# Capturar activaciones en cada paso
h1 = model_debug.layers[1](x_test)  # Dense output
h2 = model_debug.layers[2](h1)      # BatchNorm output
h3 = model_debug.layers[3](h2)      # ReLU output

println("\nEstadísticas de activaciones:")
println("  Después de Dense: mean=$(mean(h1.data)), std=$(std(h1.data))")
println("  Después de BatchNorm: mean=$(mean(h2.data)), std=$(std(h2.data))")
println("  Después de ReLU: mean=$(mean(h3.data)), std=$(std(h3.data))")
println("  Proporción de zeros después de ReLU: ", sum(h3.data .== 0) / length(h3.data))

# Test 4: Learning rate sensitivity
println("\n4️⃣ Sensibilidad al learning rate con BatchNorm")
for lr in [0.001f0, 0.01f0, 0.1f0, 1.0f0]
    model_lr = Sequential([
        Dense(2, 4),
        BatchNorm(4),
        Activation(relu),
        Dense(4, 2)
    ])
    
    opt = Adam(learning_rate=lr)
    params = collect_parameters(model_lr)
    
    initial_loss = 0.0f0
    final_loss = 0.0f0
    
    for i in 1:50
        for p in params
            zero_grad!(p)
        end
        
        pred = model_lr(X)
        loss = categorical_crossentropy(pred, y)
        
        if i == 1
            initial_loss = loss.data[1]
        elseif i == 50
            final_loss = loss.data[1]
        end
        
        backward(loss, [1.0f0])
        optim_step!(opt, params)
    end
    
    reduction = (1 - final_loss/initial_loss) * 100
    println("  LR=$lr: reducción = $(round(reduction, digits=1))%")
end

println("\n" * "="^60)