# test_batchnorm_minimal.jl

using DeepDynamics

# Test 1: Solo BatchNorm
bn = BatchNorm(16)
x = Tensor(randn(Float32, 2, 16, 8, 8); requires_grad=true)
y = bn(x)

# Backward
zero_grad!(x)
zero_grad!(bn.gamma) 
zero_grad!(bn.beta)

backward(y, ones(Float32, size(y.data)))

println("BatchNorm Test:")
println("  Input grad exists: ", x.grad !== nothing)
println("  Gamma grad exists: ", bn.gamma.grad !== nothing)
println("  Beta grad exists: ", bn.beta.grad !== nothing)

# Test 2: Conv2D + BatchNorm (como en ResNet)
println("\nConv+BatchNorm Test:")
conv = Conv2D(3, 16, (3,3), padding=(1,1))
bn2 = BatchNorm(16)

x2 = Tensor(randn(Float32, 2, 3, 8, 8))
y_conv = conv(x2)
y_bn = bn2(y_conv)

# Backward desde BatchNorm
backward(y_bn, ones(Float32, size(y_bn.data)))

println("  Conv weights grad: ", conv.weights.grad !== nothing)
println("  Conv bias grad: ", conv.bias.grad !== nothing)
println("  BN gamma grad: ", bn2.gamma.grad !== nothing)