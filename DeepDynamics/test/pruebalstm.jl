using DeepDynamics
using CUDA
using Random
using Statistics
using DeepDynamics.Optimizers: step!
using DeepDynamics.Losses: crossentropy_from_logits
using DeepDynamics
println("=== LSTM + GRU para Clasificación de Secuencias (GPU, logits CE) ===")

# ===========================
# Parámetros
# ===========================
num_samples_train = 4000
num_samples_test  = 800
seq_len   = 12
input_dim = 15
num_classes = 3

Random.seed!(1234)

println("\n📊 Generando datos sintéticos...")

function count_labels(y, num_classes)
    counts = zeros(Int, num_classes)
    for c in y
        counts[c] += 1
    end
    return counts
end

# Generar datos como (F, B, T)
X_train_data = rand(Float32, input_dim, num_samples_train, seq_len)
y_train_data = rand(1:num_classes, num_samples_train)

X_test_data  = rand(Float32, input_dim, num_samples_test, seq_len)
y_test_data  = rand(1:num_classes, num_samples_test)

println("   Train distribución:", Float32.(count_labels(y_train_data, num_classes)))
println("   Test  distribución:", Float32.(count_labels(y_test_data, num_classes)))

# One-hot encoding
function to_onehot(y, num_classes)
    Y = zeros(Float32, num_classes, length(y))
    for (i, c) in enumerate(y)
        Y[c, i] = 1f0
    end
    return Y
end

y_train_oh = to_onehot(y_train_data, num_classes)
y_test_oh  = to_onehot(y_test_data, num_classes)

# Convertir a Vector{Tensor} manteniendo (F, B, T)
X_train = [Tensor(X_train_data[:, i:i, :]) for i in 1:size(X_train_data, 2)]
y_train = [Tensor(y_train_oh[:, i:i]) for i in 1:size(y_train_oh, 2)]

X_test  = [Tensor(X_test_data[:, i:i, :]) for i in 1:size(X_test_data, 2)]
y_test  = [Tensor(y_test_oh[:, i:i]) for i in 1:size(y_test_oh, 2)]

# ===========================
# Modelo LSTM + GRU
# ===========================
println("\n🏗️ Construyendo modelo...")
model = Sequential([
    LSTM(input_dim, 32),
    GRU(32, 64),
    Flatten(),
    Dense(64 * seq_len, num_classes)
])

if CUDA.functional()
    println("   🚀 Modelo en GPU")
    model_to_gpu(model)
end

# ===========================
# Entrenamiento
# ===========================
opt = Adam(learning_rate = 0.001)

println("\n⚙️ Entrenando con fit!...")

fit!(model,
     X_train, y_train;
     X_val = X_test,
     y_val = y_test,
     loss_fn = crossentropy_from_logits,
     optimizer = opt,
     epochs = 5,
     batch_size = 32,
     metrics = [:accuracy],
     verbose = true)