# test/rnn_fit.jl
using DeepDynamics
using CUDA
using Random
Random.seed!(42)

println("=== RNN con Embedding + fit! (GPU, CCE) ===\n")

# =======================
# 1) Datos sintéticos (patrón claro)
# =======================
n_train    = 5000
n_test     = 1000
seq_len    = 10
vocab_size = 10
n_classes  = 3

println("📊 Generando datos...")
function generate_patterned_data(n_samples)
    X = zeros(Int32, seq_len, n_samples)          # (T, N) de índices 1..V
    Y = zeros(Float32, n_classes, n_samples)      # (C, N) one-hot
    for i in 1:n_samples
        class = rand(0:2)                         # 0,1,2  → clases 1..3
        seq = Int32[]
        for _ in 1:seq_len
            if rand() < 0.7
                if class == 0
                    push!(seq, rand(1:3))
                elseif class == 1
                    push!(seq, rand(4:6))
                else
                    push!(seq, rand(7:10))
                end
            else
                push!(seq, rand(1:vocab_size))
            end
        end
        X[:, i] = seq
        Y[class+1, i] = 1f0
    end
    return X, Y
end

X_train, y_train = generate_patterned_data(n_train)
X_test,  y_test  = generate_patterned_data(n_test)

train_dist = vec(sum(y_train, dims=2))
test_dist  = vec(sum(y_test,  dims=2))
println("   Train distribución: $train_dist")
println("   Test  distribución: $test_dist")
println("   Balance: $(maximum(train_dist)/minimum(train_dist) < 1.2 ? "✓" : "✗")\n")

# =======================
# 2) Modelo (Embedding → RNN → Dense → softmax)
# =======================
println("🏗️ Construyendo modelo...")

embed_dim = 32
hidden    = 64

model = Sequential([
    Embedding(vocab_size, embed_dim),                         # (T,N) índices → (T,N,E)
    RNN(embed_dim, hidden; batch_first=false,
        return_sequences=false, activation=tanh),             # (T,N,E) → (H,N) último estado
    Dense(hidden, 64),
    Activation(relu),
    Dense(64, n_classes),
    Activation(softmax)                                       # ← CCE espera probs
])

if CUDA.functional()
    model = model_to_gpu(model)
    println("   🚀 Modelo en GPU")
end

params = collect_parameters(model)
total_params = sum(length(p.data) for p in params)
println("✅ Modelo construido:")
println("   Capas: $(length(model.layers))")
println("   Parámetros totales: $total_params\n")

# =======================
# 3) DataLoaders del paquete
# =======================
# DataLoader entrega batches (X_batch, y_batch) con las mismas formas base:
#   X: (T, B) de índices 1..V   |  y: (C, B) one-hot
println("🧳 Preparando DataLoaders...")
batch_size  = 128
train_loader = DataLoader(X_train, y_train, batch_size; shuffle=true)
val_loader   = DataLoader(X_test,  y_test,  batch_size; shuffle=false)

# =======================
# 4) Callbacks y fit!
# =======================
println("\n⚙️ Entrenando con fit!...\n")

callbacks = [
    EarlyStopping(patience=8, monitor="val_loss"),
    ReduceLROnPlateau(factor=0.5f0, patience=4, monitor="val_loss"),
    ModelCheckpoint(
        "best_rnn_embed.jld2";
        monitor="val_accuracy", mode=:max, save_best_only=true
    ),
    ProgressCallback(1)
]

history = fit!(
    model, train_loader;
    val_loader=val_loader,
    epochs=40,
    optimizer=Adam(0.003f0),
    loss_fn=categorical_crossentropy,     # ← de tu Losses.jl
    callbacks=callbacks,
    verbose=1,
    log_dir="experiments",
    experiment_name="rnn_embedding_cce",
    use_tensorboard=false,
    log_config=Dict(
        "arch"        => "Embedding→RNN→Dense",
        "seq_len"     => seq_len,
        "vocab_size"  => vocab_size,
        "embed_dim"   => embed_dim,
        "hidden"      => hidden,
        "batch_size"  => batch_size,
        "optimizer"   => "Adam(0.003)"
    )
)

# =======================
# 5) Reporte rápido
# =======================
println("\n📈 Resultados:")
final_train_loss = history.train_loss[end]
final_val_loss   = history.val_loss[end]
final_train_acc  = history.train_metrics["accuracy"][end]
final_val_acc    = history.val_metrics["accuracy"][end]

println("   Loss final:   train=$(round(final_train_loss, digits=4))  |  val=$(round(final_val_loss, digits=4))")
println("   Accuracy fin: train=$(round(100*final_train_acc, digits=2))%  |  val=$(round(100*final_val_acc, digits=2))%")

# =======================
# 6) Inferencia ejemplo
# =======================
println("\n🔮 Ejemplos de predicción:")
set_training_mode!(model, false)
for _ in 1:3
    idx = rand(1:size(X_test, 2))
    seq = X_test[:, idx:idx]                  # (T,1) índices
    x   = Tensor(seq)
    if CUDA.functional(); x = to_gpu(x); end
    probs = forward(model, x).data            # (C,1)
    pred  = argmax(vec(probs))
    truth = argmax(y_test[:, idx])
    println("   Seq(head)=$(collect(seq[1:5,1])) | Real=$truth Pred=$pred (conf=$(round(100*maximum(probs),digits=1))%)")
end

println("\n" * "═"^60)
println("✅ Entrenamiento completado con fit! + CCE (probabilidades)")
