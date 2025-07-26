# test_batchnorm_gradient.jl
# Test específico para verificar el gradiente de BatchNorm

using DeepDynamics
using Test
using LinearAlgebra
using DeepDynamics.TensorEngine
println("="^60)
println("TEST ESPECÍFICO: Gradientes de BatchNorm")
println("="^60)

# Test 1: Gradiente numérico vs analítico
println("\n1️⃣ Verificación de gradiente numérico")

function numerical_gradient(f, x, eps=1e-5)
    grad = similar(x)
    fx = f(x)
    
    for i in eachindex(x)
        x_plus = copy(x)
        x_plus[i] += eps
        
        x_minus = copy(x)
        x_minus[i] -= eps
        
        grad[i] = (f(x_plus) - f(x_minus)) / (2 * eps)
    end
    
    return grad
end

# Función para evaluar BatchNorm y obtener un escalar
function bn_scalar_output(x_vec, bn)
    x = Tensor(reshape(x_vec, 2, 2, 2, 2))
    y = bn(x)
    return sum(y.data)  # Escalar para poder calcular gradiente
end

# Test con datos pequeños
bn = BatchNorm(2, training=true)
x_test = randn(Float32, 2, 2, 2, 2)
x_vec = vec(x_test)

# Gradiente numérico
f = x -> bn_scalar_output(x, bn)
grad_num = numerical_gradient(f, x_vec)

# Gradiente analítico
x_tensor = Tensor(x_test; requires_grad=true)
zero_grad!(x_tensor)
y = bn(x_tensor)
loss = sum(y.data)

loss_tensor = Tensor([loss]; requires_grad=true)
loss_tensor.backward_fn = _ -> begin
    TensorEngine.backward(y, ones(size(y.data)))
end

backward(loss_tensor, [1.0f0])
grad_analytical = vec(x_tensor.grad.data)

# Comparar
diff = norm(grad_num - grad_analytical) / (norm(grad_num) + 1e-8)
println("Diferencia relativa entre gradientes: $(round(diff, digits=6))")
println("¿Gradientes coinciden? ", diff < 0.01 ? "✅" : "❌")

# Test 2: Flujo de gradientes en cadena
println("\n2️⃣ Flujo de gradientes a través de capas")

x = Tensor(randn(Float32, 4, 2, 4, 4); requires_grad=true)
bn = BatchNorm(2, training=true)
dense = Dense(2*4*4, 1)

# Forward
h1 = bn(x)
h1_flat = Flatten()(h1)
h2 = dense(h1_flat)
loss = sum(h2.data)

# Zero grads
zero_grad!(x)
zero_grad!(bn.gamma)
zero_grad!(bn.beta) 
zero_grad!(dense.weights)
zero_grad!(dense.biases)

# Crear loss tensor
loss_tensor = Tensor([loss]; requires_grad=true)
loss_tensor.backward_fn = _ -> begin
    TensorEngine.backward(h2, ones(size(h2.data)))
end

# Backward
backward(loss_tensor, [1.0f0])

println("\nGradientes después de backward:")
println("  x.grad existe: ", x.grad !== nothing)
println("  bn.gamma.grad existe: ", bn.gamma.grad !== nothing)
println("  bn.beta.grad existe: ", bn.beta.grad !== nothing)
println("  dense.weights.grad existe: ", dense.weights.grad !== nothing)

if x.grad !== nothing
    println("  Norma grad x: ", norm(x.grad.data))
    println("  Max abs grad x: ", maximum(abs.(x.grad.data)))
    println("  Min abs grad x: ", minimum(abs.(x.grad.data)))
end

# Test 3: Caso mínimo que reproduce el problema
println("\n3️⃣ Caso mínimo de no convergencia")

# Datos XOR (no linealmente separable, necesita capas ocultas)
X = Tensor(Float32[0 0 1 1; 0 1 0 1])
y = Tensor(Float32[0 1 1 0; 1 0 0 1])

println("\nSin BatchNorm:")
model1 = Sequential([
    Dense(2, 4),
    Activation(relu),
    Dense(4, 2)
])

losses1 = Float32[]
opt1 = SGD(learning_rate=0.5f0)
params1 = collect_parameters(model1)

for i in 1:50
    for p in params1
        zero_grad!(p)
    end
    
    pred = model1(X)
    loss = categorical_crossentropy(pred, y)
    push!(losses1, loss.data[1])
    
    backward(loss, [1.0f0])
    optim_step!(opt1, params1)
end

println("  Primeras 5 losses: ", losses1[1:5])
println("  Últimas 5 losses: ", losses1[end-4:end])

println("\nCon BatchNorm:")
model2 = Sequential([
    Dense(2, 4),
    BatchNorm(4),
    Activation(relu),
    Dense(4, 2)
])

losses2 = Float32[]
opt2 = SGD(learning_rate=0.5f0)
params2 = collect_parameters(model2)

# Verificar que BatchNorm está en params
bn_in_params = any(p -> p === model2.layers[2].gamma || p === model2.layers[2].beta, params2)
println("  ¿BatchNorm params en collect_parameters? ", bn_in_params)

for i in 1:50
    for p in params2
        zero_grad!(p)
    end
    
    pred = model2(X)
    loss = categorical_crossentropy(pred, y)
    push!(losses2, loss.data[1])
    
    backward(loss, [1.0f0])
    
    # Debug en primera iteración
    if i == 1
        println("\n  Debug primera iteración:")
        for (j, layer) in enumerate(model2.layers)
            if layer isa Dense
                println("    Dense $j - grad weights norm: ", norm(layer.weights.grad.data))
            elseif layer isa BatchNorm
                println("    BatchNorm - grad gamma norm: ", norm(layer.gamma.grad.data))
                println("    BatchNorm - grad beta norm: ", norm(layer.beta.grad.data))
            end
        end
    end
    
    optim_step!(opt2, params2)
end

println("\n  Primeras 5 losses: ", losses2[1:5])
println("  Últimas 5 losses: ", losses2[end-4:end])

println("\n" * "="^60)