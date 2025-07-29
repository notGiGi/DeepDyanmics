# test_verify_fixes.jl

using DeepDynamics
using DeepDynamics.Layers
using DeepDynamics.ConvolutionalLayers
println("=== Verificando fixes de GlobalAvgPool y MaxPooling ===")

# Test 1: GlobalAvgPool con el fix
println("\n1. GlobalAvgPool (con fix):")
x1 = Tensor(randn(Float32, 2, 16, 8, 8); requires_grad=true)
gap = GlobalAvgPool()
y1 = gap(x1)
zero_grad!(x1)
backward(y1, ones(Float32, size(y1.data)))
println("   Output shape: ", size(y1.data))
println("   Input grad exists: ", x1.grad !== nothing)
if x1.grad !== nothing
    println("   Input grad shape: ", size(x1.grad.data))
    println("   Input grad non-zero: ", any(x1.grad.data .!= 0))
    println("   Input grad sum: ", sum(x1.grad.data))
end

# Test 2: MaxPooling con el fix
println("\n2. MaxPooling (con fix):")
x2 = Tensor(randn(Float32, 2, 16, 8, 8); requires_grad=true)
pool = MaxPooling((2,2))
y2 = pool(x2)
zero_grad!(x2)
backward(y2, ones(Float32, size(y2.data)))
println("   Output shape: ", size(y2.data))
println("   Input grad exists: ", x2.grad !== nothing)
if x2.grad !== nothing
    println("   Input grad non-zero: ", any(x2.grad.data .!= 0))
    println("   Non-zero count: ", count(x2.grad.data .!= 0))
end

# Test 3: Cadena completa
println("\n3. Cadena completa (Conv → MaxPool → GAP):")
conv = Conv2D(3, 16, (3,3), padding=(1,1))
x3 = Tensor(randn(Float32, 2, 3, 8, 8); requires_grad=true)

y_conv = conv(x3)
y_pool = MaxPooling((2,2))(y_conv)
y_gap = GlobalAvgPool()(y_pool)

zero_grad!(conv.weights)
zero_grad!(conv.bias)
backward(y_gap, ones(Float32, size(y_gap.data)))

println("   Conv weights grad exists: ", conv.weights.grad !== nothing)
println("   Conv bias grad exists: ", conv.bias.grad !== nothing)
if conv.weights.grad !== nothing
    println("   Conv weights grad non-zero: ", any(conv.weights.grad.data .!= 0))
end