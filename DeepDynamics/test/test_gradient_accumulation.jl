    using DeepDynamics
using LinearAlgebra

println("=== Test de Acumulación de Gradientes ===")

model = Sequential([Dense(2, 1)])
params = collect_parameters(model)

X = Tensor(randn(Float32, 2, 5))
y = Tensor(randn(Float32, 1, 5))

println("\n1. Test SIN zero_grad! (problema):")
for i in 1:3
    output = model(X)
    loss = mse_loss(output, y)
    backward(loss, Tensor([1.0f0]))
    println("  Iteración $i, norm grad: ", norm(params[1].grad.data))
    end

model = Sequential([Dense(2, 1)])
params = collect_parameters(model)

println("\n2. Test CON zero_grad! (solución):")
for i in 1:3
    for p in params
        zero_grad!(p)
    end
    output = model(X)
    loss = mse_loss(output, y)
    backward(loss, Tensor([1.0f0]))
        println("  Iteración $i, norm grad: ", norm(params[1].grad.data))
        end
        
        println("\n✓ Con zero_grad!, los gradientes NO se acumulan entre iteraciones")
