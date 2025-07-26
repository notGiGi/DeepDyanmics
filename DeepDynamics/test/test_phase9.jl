println("📦 Running Fase 9 Test – ResNet Support")

using DeepDynamics
using DeepDynamics.TensorEngine
using DeepDynamics.NeuralNetwork
using DeepDynamics.Models: create_resnet
using DeepDynamics.Optimizers
using DeepDynamics.Losses
using Statistics
using CUDA

# ---------------------------
# FUNCIONES AUXILIARES
# ---------------------------

function verificar_gradientes(params)
    grad_count = 0
    no_grad_params = Int[]
    for (i, p) in enumerate(params)
        if p.requires_grad && p.grad !== nothing && any(p.grad.data .!= 0)
            grad_count += 1
        else
            push!(no_grad_params, i)
        end
    end

    if grad_count == length(params)
        println("✅ Todos los parámetros recibieron gradientes ($grad_count/$(length(params)))")
    else
        println("❌ Solo $grad_count/$(length(params)) parámetros recibieron gradientes")
        if length(no_grad_params) <= 10
            println("   Sin gradientes: ", no_grad_params)
        else
            println("   Sin gradientes: ", no_grad_params[1:5], "...", no_grad_params[end-4:end])
        end
    end
end

# ---------------------------
# 1. Generar datos aprendibles
# ---------------------------

batch_size = 16
num_classes = 10
input_channels = 3
input_size = 64

X = zeros(Float32, batch_size, input_channels, input_size, input_size)
y = zeros(Float32, num_classes, batch_size)

for i in 1:batch_size
    clase = rand(1:num_classes)
    X[i, 1, :, :] .= clase / num_classes
    X[i, 2, :, :] .= rand(Float32) * 0.1
    X[i, 3, :, :] .= rand(Float32) * 0.1
    y[clase, i] = 1.0f0
end

println("Tamaño de X: ", size(X))  # (16, 3, 64, 64)

# ---------------------------
# 2. Crear modelo ResNet
# ---------------------------

model = create_resnet(input_channels, num_classes, blocks=[2,2])
println("✅ Modelo ResNet creado exitosamente.")

# ---------------------------
# 3. Mover a GPU si es posible
# ---------------------------

if CUDA.functional()
    X = CUDA.CuArray(X)
    y = CUDA.CuArray(y)
    model = DeepDynamics.model_to_gpu(model)
end

# ---------------------------
# 4. Parámetros del modelo
# ---------------------------

params = collect_parameters(model)
println("Total parámetros: $(length(params))")
for (i, p) in enumerate(params)
    println("Param $i → requires_grad: ", p.requires_grad)
end

# ---------------------------
# 5. Entrenamiento
# ---------------------------

opt = Adam(learning_rate=0.001)
println("🚀 Entrenando por 5 épocas")

loss_history = Float32[]

for epoch in 1:5
    NeuralNetwork.zero_grad!(model)

    x_tensor = Tensor(X; requires_grad=false)
    y_tensor = Tensor(y; requires_grad=false)

    ŷ = model(x_tensor)
    loss = categorical_crossentropy(ŷ, y_tensor)

    backward(loss, [1.0f0])

    if epoch == 1
        println("\n📊 Verificación de gradientes:")
        grad_count = 0
        for (i, p) in enumerate(params)
            if p.grad !== nothing && any(p.grad.data .!= 0)
                grad_count += 1
                if i <= 5 || i > length(params) - 3
                    max_grad = maximum(abs.(p.grad.data))
                    println("✅ Param $i con gradiente, max: $max_grad")
                end
            else
                if i <= 5 || i > length(params) - 3
                    println("❌ Param $i sin gradiente")
                end
            end
        end
        println("Total con gradientes: $grad_count/$(length(params))")
    end

    DeepDynamics.optim_step!(opt, params)
    push!(loss_history, loss.data[1])
    println("📉 Epoch $epoch – Loss = $(round(loss.data[1], digits=4))")
end

# ---------------------------
# 6. Validaciones finales
# ---------------------------

println("\n📋 Validaciones finales:")

if loss_history[end] < loss_history[1]
    println("✅ Loss disminuyó: $(loss_history[1]) → $(loss_history[end])")
else
    println("❌ Loss no disminuyó: $(loss_history[1]) → $(loss_history[end])")
end

# Verificación independiente de gradientes post-entrenamiento
NeuralNetwork.zero_grad!(model)
x_test = Tensor(X; requires_grad=false)
y_test = Tensor(y; requires_grad=false)
out_test = model(x_test)
loss_test = categorical_crossentropy(out_test, y_test)
backward(loss_test, [1.0f0])
verificar_gradientes(params)

# Verificar tamaño de salida
test_output = model(Tensor(X))
if size(test_output.data) == (num_classes, batch_size)
    println("✅ Dimensiones de salida correctas")
else
    println("❌ Dimensiones incorrectas: $(size(test_output.data))")
end

println("\n✅ Fase 9 Test COMPLETADO")
