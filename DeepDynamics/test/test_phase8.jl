using DeepDynamics
using CUDA
using DeepDynamics.GPUMemoryManager
using DeepDynamics.TensorEngine
using DeepDynamics.NeuralNetwork
println("=== TEST FASE 8: Mejoras de Estabilidad (ACTUALIZADO) ===")
using DeepDynamics.TensorEngine: ensure_on_device

# Test 1: set_training_mode!
println("\nTest 1: set_training_mode!")
model = Sequential([
    Dense(10, 20),
    BatchNorm(20),
    Activation(relu),
    DropoutLayer(0.5),
    Dense(20, 10)
])

bn_layer = model.layers[2]
dropout_layer = model.layers[4]
println("BatchNorm training inicial: ", bn_layer.training)
println("Dropout training inicial: ", dropout_layer.training)

set_training_mode!(model, false)
println("Después de set_training_mode!(model, false):")
println("  BatchNorm: ", bn_layer.training)
println("  Dropout: ", dropout_layer.training)
println("  BatchNorm tracked batches: ", bn_layer.num_batches_tracked)
@assert !bn_layer.training "BatchNorm debería estar en modo eval"
@assert !dropout_layer.training "Dropout debería estar en modo eval"

set_training_mode!(model, true)
@assert bn_layer.training "BatchNorm debería estar en modo training"
@assert dropout_layer.training "Dropout debería estar en modo training"
println("✓ set_training_mode! funciona correctamente")

# Test 2: Dropout estabilidad
println("\nTest 2: Dropout estabilidad")
x = Tensor(randn(Float32, 10, 10); requires_grad=true)
dropout = DropoutLayer(0.5, true)

out_train = forward(dropout, x)
@assert size(out_train.data) == size(x.data)
@assert out_train.requires_grad == x.requires_grad

num_zeros = count(out_train.data .== 0)
@assert num_zeros > 0 "Debe haber elementos dropeados en training"
println("  Elementos dropeados: $num_zeros de $(length(out_train.data))")

set_training_mode!(dropout, false)
out_eval = forward(dropout, x)
@assert all(out_eval.data .== x.data)
println("✓ Dropout funciona correctamente en ambos modos")

# Test 3: Dropout backward
println("\nTest 3: Dropout backward")
dropout_train = DropoutLayer(0.5, true)
x_grad = Tensor(ones(Float32, 5, 5); requires_grad=true)
out = forward(dropout_train, x_grad)
if out.backward_fn !== nothing
    out.backward_fn(ones(Float32, size(out.data)))
    @assert x_grad.grad !== nothing "Debe tener gradiente"
    println("  Gradiente propagado correctamente")
end
println("✓ Dropout backward funciona")

# Test 4: GPU tests
if CUDA.functional()
    println("\nTest 4: Tests GPU")
    x_gpu = to_gpu(Tensor(randn(Float32, 100, 100); requires_grad=true))
    dropout_gpu = DropoutLayer(0.3, true)
    out_gpu = forward(dropout_gpu, x_gpu)

    @assert out_gpu.data isa CUDA.CuArray
    @assert size(out_gpu.data) == size(x_gpu.data)

    num_zeros = count(Array(out_gpu.data) .== 0)
    total_elements = length(out_gpu.data)
    dropout_ratio = num_zeros / total_elements
    println("  Ratio de dropout observado: ", round(dropout_ratio, digits=3))
    @assert 0.2 < dropout_ratio < 0.4

    stats = GPUMemoryManager.memory_stats()
    println("  Memoria GPU - Total: $(round(stats.total, digits=2)) GB")
    println("  Memoria GPU - Usada: $(round(stats.used, digits=2)) GB")
    println("  Memoria GPU - Libre: $(round(stats.free_percent, digits=1))%")
    GPUMemoryManager.check_and_clear_gpu_memory(verbose=true)
    println("✓ Funcionalidad GPU correcta")
else
    println("\nTest 4: GPU no disponible, saltando tests GPU")
end

# Test 5: BatchNorm con tracking
println("\nTest 5: BatchNorm tracking")
bn = BatchNorm(10)
x_bn = Tensor(randn(Float32, 2, 10, 5, 5); requires_grad=true)
set_training_mode!(bn, true)
initial_batches = bn.num_batches_tracked
_ = forward(bn, x_bn)
@assert bn.num_batches_tracked == initial_batches + 1

set_training_mode!(bn, false)
_ = forward(bn, x_bn)
@assert bn.num_batches_tracked == initial_batches + 1
println("✓ BatchNorm tracking funciona correctamente")

# Test 6: Integración completa
println("\nTest 6: Integración con train_improved!")
X = Tensor(randn(Float32, 10, 50); requires_grad=false)
y = Tensor(randn(Float32, 5, 50); requires_grad=false)

model = Sequential([
    Dense(10, 20),
    DropoutLayer(0.5, true),
    BatchNorm(20),
    Activation(relu),
    Dense(20, 5)
])
bn_layer = model.layers[3]
println("  BN init running_var: ", bn_layer.running_var)
println("  BN init epsilon: ", bn_layer.epsilon)

println("  X stats: min=$(minimum(X.data)), max=$(maximum(X.data)), any NaN=$(any(isnan.(X.data)))")
println("  y stats: min=$(minimum(y.data)), max=$(maximum(y.data)), any NaN=$(any(isnan.(y.data)))")

opt = Adam(learning_rate=0.01)
dropout_layer = model.layers[2]
bn_layer = model.layers[3]
println("  Estado inicial - Dropout: $(dropout_layer.training), BN: $(bn_layer.training)")

# 🔧 NUEVO: Asegurar que X, y estén en el mismo dispositivo que el modelo
device = model_device(model)
X = ensure_on_device(X, device)
y = ensure_on_device(y, device)

# Entrenamiento
losses = train_improved!(model, opt, mse_loss, [X], [y], 1;
                        batch_size=10, verbose=true)

train_losses = losses[1]
@assert !isempty(train_losses) "No se registraron pérdidas"
@assert !isnan(train_losses[end]) "Loss final no debe ser NaN"
println("  Loss final: ", train_losses[end])

@assert length(losses) > 0 "Debe haber losses registrados"

println("  Estado final - Dropout: $(dropout_layer.training), BN: $(bn_layer.training)")
@assert !dropout_layer.training
@assert !bn_layer.training

println("✓ Integración completa exitosa")
println("\n=== TODOS LOS TESTS DE FASE 8 PASARON ===")
