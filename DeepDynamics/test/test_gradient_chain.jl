# test_gradient_chain.jl - Encuentra dónde se cortan los gradientes

using DeepDynamics

println("=== Test de propagación de gradientes por capa ===")

# Test 1: GlobalAvgPool
println("\n1. GlobalAvgPool:")
x1 = Tensor(randn(Float32, 2, 16, 8, 8); requires_grad=true)
gap = GlobalAvgPool()
y1 = gap(x1)
zero_grad!(x1)
backward(y1, ones(Float32, size(y1.data)))
println("   Input grad exists: ", x1.grad !== nothing)
if x1.grad !== nothing
    println("   Input grad non-zero: ", any(x1.grad.data .!= 0))
end

# Test 2: Flatten
println("\n2. Flatten:")
x2 = Tensor(randn(Float32, 2, 16, 8, 8); requires_grad=true)
flat = Flatten()
y2 = flat(x2)
zero_grad!(x2)
backward(y2, ones(Float32, size(y2.data)))
println("   Input grad exists: ", x2.grad !== nothing)
if x2.grad !== nothing
    println("   Input grad non-zero: ", any(x2.grad.data .!= 0))
end

# Test 3: DropoutLayer
println("\n3. DropoutLayer:")
x3 = Tensor(randn(Float32, 2, 128); requires_grad=true)
drop = DropoutLayer(0.5f0)
y3 = drop(x3)
zero_grad!(x3)
backward(y3, ones(Float32, size(y3.data)))
println("   Input grad exists: ", x3.grad !== nothing)
if x3.grad !== nothing
    println("   Input grad non-zero: ", any(x3.grad.data .!= 0))
end

# Test 4: MaxPooling
println("\n4. MaxPooling:")
x4 = Tensor(randn(Float32, 2, 16, 8, 8); requires_grad=true)
pool = MaxPooling((2,2))
y4 = pool(x4)
zero_grad!(x4)
backward(y4, ones(Float32, size(y4.data)))
println("   Input grad exists: ", x4.grad !== nothing)
if x4.grad !== nothing
    println("   Input grad non-zero: ", any(x4.grad.data .!= 0))
end

# Test 5: LayerActivation(relu)
println("\n5. LayerActivation(relu):")
x5 = Tensor(randn(Float32, 2, 16, 8, 8); requires_grad=true)
act = LayerActivation(relu)
y5 = act(x5)
zero_grad!(x5)
backward(y5, ones(Float32, size(y5.data)))
println("   Input grad exists: ", x5.grad !== nothing)
if x5.grad !== nothing
    println("   Input grad non-zero: ", any(x5.grad.data .!= 0))
end

# Test 6: Cadena completa hasta Dense
println("\n6. Cadena GAP → Flatten → Dropout → Dense:")
x6 = Tensor(randn(Float32, 2, 64, 8, 8); requires_grad=true)
# Aplicar capas en secuencia
y_gap = GlobalAvgPool()(x6)
y_flat = Flatten()(y_gap)
y_drop = DropoutLayer(0.5f0)(y_flat)
dense = Dense(64, 10)
y_dense = dense(y_drop)

# Backward
zero_grad!(x6)
zero_grad!(dense.weights)
zero_grad!(dense.biases)
backward(y_dense, ones(Float32, size(y_dense.data)))

println("   Input grad exists: ", x6.grad !== nothing)
println("   Dense weights grad exists: ", dense.weights.grad !== nothing)
if x6.grad !== nothing
    println("   Input grad non-zero: ", any(x6.grad.data .!= 0))
end