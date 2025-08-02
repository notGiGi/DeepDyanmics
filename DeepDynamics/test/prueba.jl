# test_fit_with_logging.jl
using DeepDynamics
using CUDA

println("=== Test de fit! con Sistema de Logging ===\n")

# 1. Crear datos sintéticos
println("1️⃣ Generando datos sintéticos...")
n_samples = 1000
n_features = 10
n_classes = 3

# Generar datos de clasificación multiclase
X_data = randn(Float32, n_features, n_samples)
y_data = zeros(Float32, n_classes, n_samples)
for i in 1:n_samples
    class = rand(1:n_classes)
    y_data[class, i] = 1.0f0
end

# Convertir a tensores
X = [Tensor(X_data[:, i:i]) for i in 1:n_samples]
y = [Tensor(y_data[:, i:i]) for i in 1:n_samples]

println("✓ Datos generados: $n_samples muestras, $n_features features, $n_classes clases")

# 2. Crear modelo
println("\n2️⃣ Creando modelo...")
model = Sequential([
    Dense(n_features, 64),
    BatchNorm(64),
    Activation(relu),
    DropoutLayer(0.5),
    Dense(64, 32),
    BatchNorm(32),
    Activation(relu),
    Dense(32, n_classes),
    Activation(softmax)
])

# Mover a GPU si está disponible
if CUDA.functional()
    model = model_to_gpu(model)
    println("✓ Modelo movido a GPU")
else
    println("✓ Modelo en CPU")
end

# 3. Configurar callbacks adicionales
println("\n3️⃣ Configurando callbacks...")
callbacks = [
    EarlyStopping(patience=5, monitor="val_loss"),
    ReduceLROnPlateau(factor=0.5f0, patience=3),
    ModelCheckpoint("best_model.jld2", monitor="val_accuracy", mode="max")
]

# 4. Entrenar SIN logging (baseline)
println("\n4️⃣ Entrenamiento baseline (sin logging)...")
history_baseline = fit!(
    model, X[1:800], y[1:800],
    validation_data=(X[801:end], y[801:end]),
    epochs=10,
    batch_size=32,
    optimizer=Adam(0.001f0),
    loss_fn=categorical_crossentropy,
    callbacks=callbacks,
    verbose=true
)

println("\nResultados baseline:")
println("  Loss final: $(history_baseline.train_loss[end])")
println("  Val loss final: $(history_baseline.val_loss[end])")

# 5. Entrenar CON logging completo
println("\n5️⃣ Entrenamiento con logging completo...")

# Reinicializar modelo
model = Sequential([
    Dense(n_features, 64),
    BatchNorm(64),
    Activation(relu),
    DropoutLayer(0.5),
    Dense(64, 32),
    BatchNorm(32),
    Activation(relu),
    Dense(32, n_classes),
    Activation(softmax)
])

if CUDA.functional()
    model = model_to_gpu(model)
end

# Entrenar con logging
history_logged = fit!(
    model, X[1:800], y[1:800],
    validation_data=(X[801:end], y[801:end]),
    epochs=20,
    batch_size=32,
    optimizer=Adam(0.001f0),
    loss_fn=categorical_crossentropy,
    callbacks=callbacks,
    verbose=true,
    # PARÁMETROS DE LOGGING
    log_dir="experiments",
    experiment_name="test_multiclass_classification",
    use_tensorboard=true,
    log_gradients=true,
    log_config=Dict(
        "dataset" => "synthetic",
        "architecture" => "MLP_with_BN",
        "regularization" => "dropout_0.5",
        "notes" => "Testing logging integration"
    )
)

println("\n✅ Entrenamiento completado con logging!")

# 6. Verificar archivos creados
exp_dir = "experiments"
if isdir(exp_dir)
    println("\n6️⃣ Archivos de logging creados:")
    for (root, dirs, files) in walkdir(exp_dir)
        for file in files
            rel_path = relpath(joinpath(root, file), exp_dir)
            println("  📄 $rel_path")
        end
    end
end

# 7. Comparar experimentos
println("\n7️⃣ Comparando experimentos...")
experiments = readdir(exp_dir)
if length(experiments) >= 2
    exp_ids = experiments[end-1:end]  # Últimos 2
    comparisons = compare_experiments(exp_ids, exp_dir)
    
    println("\nComparación de últimos 2 experimentos:")
    for comp in comparisons
        println("\n  Experimento: $(comp["experiment_id"])")
        if haskey(comp, "final_metrics") && haskey(comp["final_metrics"], "loss")
            println("    Loss final: $(comp["final_metrics"]["loss"])")
        end
    end
end

# 8. Test con DataLoader
println("\n8️⃣ Test con DataLoader...")
train_loader = DataLoader(X[1:800], y[1:800], 32; shuffle=true)
val_loader = DataLoader(X[801:end], y[801:end], 32; shuffle=false)

history_dataloader = fit!(
    model, train_loader,
    val_loader=val_loader,
    epochs=5,
    optimizer=Adam(0.0001f0),
    loss_fn=categorical_crossentropy,
    verbose=1,
    log_dir="experiments",
    experiment_name="test_with_dataloader",
    use_tensorboard=true
)

println("\n✅ Test con DataLoader completado!")

# 9. Análisis de métricas
println("\n9️⃣ Resumen de métricas:")
println("  Entrenamientos realizados: 3")
println("  Mejor val_loss baseline: $(minimum(history_baseline.val_loss))")
println("  Mejor val_loss con logging: $(minimum(history_logged.val_loss))")
println("  Archivos de log generados: $(length(readdir(exp_dir)))")

# 10. Ejemplo de carga de logs
println("\n🔟 Ejemplo de lectura de logs:")
latest_exp = readdir(exp_dir)[end]
log_file = joinpath(exp_dir, latest_exp, "training.jsonl")
if isfile(log_file)
    # Leer primeras 3 líneas
    lines = readlines(log_file)[1:min(3, end)]
    for (i, line) in enumerate(lines)
        data = JSON.parse(line)
        println("  Línea $i: evento '$(get(data, "event", "?"))' en t=$(round(get(data, "time_elapsed", 0), digits=2))s")
    end
end

println("\n🎉 ¡Todos los tests completados exitosamente!")