# test_conv2d_minimal.jl - Test aislado de Conv2D

using DeepDynamics

# Crear una Conv2D simple
conv = Conv2D(3, 16, (3,3), stride=(1,1), padding=(1,1))

# Input pequeño
x = Tensor(randn(Float32, 2, 3, 8, 8); requires_grad=true)

# Forward
y = conv(x)

# Loss simple
loss = sum(y.data)

# Backward
zero_grad!(conv.weights)
zero_grad!(conv.bias)
zero_grad!(x)

backward(y, ones(Float32, size(y.data)))

# Verificar gradientes
println("Conv2D weights grad exists: ", conv.weights.grad !== nothing)
println("Conv2D bias grad exists: ", conv.bias.grad !== nothing)
println("Input grad exists: ", x.grad !== nothing)

if conv.weights.grad !== nothing
    println("Weights grad non-zero: ", any(conv.weights.grad.data .!= 0))
end
if conv.bias.grad !== nothing
    println("Bias grad non-zero: ", any(conv.bias.grad.data .!= 0))
end