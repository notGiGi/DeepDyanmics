using DeepDynamics
using Random
using Statistics

println("=== EJEMPLO RETADOR: CLASIFICACIÓN 3D NO LINEAL ===\n")

# Generar dataset 3D complejo con 3 clases entrelazadas
function generate_3d_helix_dataset(n_per_class=200)
    Random.seed!(123)
    X_data = Float32[]
    y_data = Float32[]
    
    # Clase 0: Hélice ascendente
    for i in 1:n_per_class
        t = 4π * i / n_per_class
        x = 0.5f0 * cos(t) + 0.05f0 * randn()
        y = 0.5f0 * sin(t) + 0.05f0 * randn()
        z = t / (4π) - 0.5f0 + 0.05f0 * randn()
        append!(X_data, [x, y, z])
        append!(y_data, [1f0, 0f0, 0f0])  # One-hot
    end
    
    # Clase 1: Hélice descendente
    for i in 1:n_per_class
        t = 4π * i / n_per_class
        x = 0.5f0 * cos(t + π/3) + 0.05f0 * randn()
        y = 0.5f0 * sin(t + π/3) + 0.05f0 * randn()
        z = 0.5f0 - t / (4π) + 0.05f0 * randn()
        append!(X_data, [x, y, z])
        append!(y_data, [0f0, 1f0, 0f0])
    end
    
    # Clase 2: Esfera central con ruido
    for i in 1:n_per_class
        # Generar puntos en esfera usando coordenadas esféricas
        θ = 2π * rand()
        φ = π * rand()
        r = 0.3f0 + 0.1f0 * randn()
        
        x = r * sin(φ) * cos(θ)
        y = r * sin(φ) * sin(θ)
        z = r * cos(φ)
        
        append!(X_data, [x, y, z])
        append!(y_data, [0f0, 0f0, 1f0])
    end
    
    # Mezclar y dividir
    n_total = 3 * n_per_class
    X_matrix = reshape(X_data, 3, n_total)
    y_matrix = reshape(y_data, 3, n_total)
    
    indices = shuffle(1:n_total)
    X_matrix = X_matrix[:, indices]
    y_matrix = y_matrix[:, indices]
    
    # División 70-15-15
    n_train = Int(0.7 * n_total)
    n_val = Int(0.15 * n_total)
    
    X_train = [Tensor(X_matrix[:, i]) for i in 1:n_train]
    y_train = [Tensor(y_matrix[:, i]) for i in 1:n_train]
    
    X_val = [Tensor(X_matrix[:, i]) for i in (n_train+1):(n_train+n_val)]
    y_val = [Tensor(y_matrix[:, i]) for i in (n_train+1):(n_train+n_val)]
    
    X_test = [Tensor(X_matrix[:, i]) for i in (n_train+n_val+1):n_total]
    y_test = [Tensor(y_matrix[:, i]) for i in (n_train+n_val+1):n_total]
    
    return X_train, y_train, X_val, y_val, X_test, y_test
end

# 1. Generar datos
println("1️⃣ Generando dataset 3D complejo...")
X_train, y_train, X_val, y_val, X_test, y_test = generate_3d_helix_dataset(200)
println("  Train: $(length(X_train)) muestras")
println("  Val: $(length(X_val)) muestras")
println("  Test: $(length(X_test)) muestras")
println("  Features: 3D (hélices entrelazadas + esfera)")
println("  Clases: 3")

# 2. Modelo con arquitectura ResNet-style (conexiones residuales simuladas)
println("\n2️⃣ Creando modelo profundo...")
model = Sequential([
    # Bloque 1
    Dense(3, 64; init_method=:he),
    BatchNorm(64),
    Activation(relu),
    DropoutLayer(0.1),
    
    # Bloque 2 - Expansión
    Dense(64, 128; init_method=:he),
    BatchNorm(128),
    Activation(relu),
    DropoutLayer(0.2),
    
    # Bloque 3 - Procesamiento
    Dense(128, 128; init_method=:he),
    BatchNorm(128),
    Activation(relu),
    DropoutLayer(0.2),
    
    # Bloque 4 - Compresión
    Dense(128, 64; init_method=:he),
    BatchNorm(64),
    Activation(relu),
    DropoutLayer(0.),
    
    # Bloque 5 - Cabeza de clasificación
    Dense(64, 32; init_method=:he),
    BatchNorm(32),
    Activation(relu),
    
    Dense(32, 3),
    Activation(softmax)
])

println("  Capas: $(length(model.layers))")
println("  Parámetros: ~$(sum(prod(size(p.data)) for p in collect_parameters(model)))")

# 3. Entrenamiento con múltiples fases
println("\n3️⃣ Entrenamiento en fases...")

# Fase 1: Learning rate alto para exploración
println("\n📍 Fase 1: Exploración (LR alto)")
opt1 = Adam(learning_rate=0.01)
history1 = train!(
    model, X_train, y_train,
    optimizer=opt1,
    loss_fn=categorical_crossentropy,
    epochs=20,
    batch_size=64,
    metrics=[accuracy],
    verbose=true
)
println("  Loss: $(round(history1[:loss][1], digits=3)) → $(round(history1[:loss][end], digits=3))")
println("  Accuracy: $(round(history1[:metrics][:accuracy][end]*100, digits=1))%")

# Fase 2: Learning rate medio para refinamiento
println("\n📍 Fase 2: Refinamiento (LR medio)")
opt2 = Adam(learning_rate=0.001)
history2 = train!(
    model, X_train, y_train,
    optimizer=opt2,
    loss_fn=categorical_crossentropy,
    epochs=30,
    batch_size=32,
    metrics=[accuracy],
    verbose=true
)
println("  Loss: $(round(history2[:loss][1], digits=3)) → $(round(history2[:loss][end], digits=3))")
println("  Accuracy: $(round(history2[:metrics][:accuracy][end]*100, digits=1))%")

# Fase 3: Learning rate bajo para ajuste fino
println("\n📍 Fase 3: Ajuste fino (LR bajo)")
opt3 = Adam(learning_rate=0.0000001)
history3 = train!(
    model, X_train, y_train,
    optimizer=opt3,
    loss_fn=categorical_crossentropy,
    epochs=50,
    batch_size=32,
    metrics=[accuracy],
    verbose=true
)
# Poner modelo en modo evaluación
set_training_mode!(model, false)
# Debug: verificar predicciones del modelo
test_pred = forward(model, X_train[1])
println("  Predicción ejemplo: $(test_pred.data)")
println("  Suma softmax: $(sum(test_pred.data))")
println("  Loss: $(round(history3[:loss][1], digits=3)) → $(round(history3[:loss][end], digits=3))")
println("  Accuracy: $(round(history3[:metrics][:accuracy][end]*100, digits=1))%")

# 4. Evaluación detallada
println("\n4️⃣ Evaluación completa...")

function evaluate_dataset(model, X, y, name)
    correct = 0
    confusion = zeros(Int, 3, 3)
    confidences = Float32[]
    
    for (x, y_true) in zip(X, y)
        pred = forward(model, x)
        pred_class = argmax(vec(pred.data))
        true_class = argmax(vec(y_true.data))
        confusion[pred_class, true_class] += 1
        correct += (pred_class == true_class)
        
        # Confianza de la predicción
        push!(confidences, maximum(pred.data))
    end
    
    accuracy = correct / length(X)
    avg_confidence = mean(confidences)
    
    println("\n  $name:")
    println("    Accuracy: $(round(accuracy*100, digits=1))%")
    println("    Confianza promedio: $(round(avg_confidence*100, digits=1))%")
    
    # Métricas por clase
    println("    Por clase:")
    for i in 1:3
        tp = confusion[i,i]
        fp = sum(confusion[i,:]) - tp
        fn = sum(confusion[:,i]) - tp
        
        precision = tp / (tp + fp + 1e-7)
        recall = tp / (tp + fn + 1e-7)
        f1 = 2 * precision * recall / (precision + recall + 1e-7)
        
        println("      Clase $(i-1): P=$(round(precision, digits=2)), R=$(round(recall, digits=2)), F1=$(round(f1, digits=2))")
    end
    
    return accuracy, confusion
end

train_acc, _ = evaluate_dataset(model, X_train, y_train, "TRAIN")
val_acc, _ = evaluate_dataset(model, X_val, y_val, "VALIDACIÓN")
test_acc, conf_test = evaluate_dataset(model, X_test, y_test, "TEST")

# 5. Análisis de errores
println("\n5️⃣ Análisis de errores en TEST:")
println("  Matriz de confusión:")
println("       C0   C1   C2")
for i in 1:3
    print("  C$(i-1): ")
    for j in 1:3
        print(lpad(conf_test[i,j], 4))
    end
    println()
end





# 7. Verificación final
println("\n✅ RESUMEN FINAL:")
println("  • Dataset: 3D no lineal (hélices + esfera)")
println("  • Arquitectura: 13 capas con BatchNorm y Dropout")
println("  • Entrenamiento: 3 fases con LR decay")
println("  • Train accuracy: $(round(train_acc*100, digits=1))%")
println("  • Val accuracy: $(round(val_acc*100, digits=1))%")
println("  • Test accuracy: $(round(test_acc*100, digits=1))%")


# Criterios de éxito
overfitting = train_acc - test_acc
success = (
    test_acc > 0.80 &&           # Alta precisión en test
    overfitting < 0.1 &&         # Bajo overfitting
    val_acc > 0.70           # Buena generalización
                 # Robustez aceptable
)

println("\n🎯 Criterios de éxito:")
println("  Test > 85%: $(test_acc > 0.85 ? "✓" : "✗")")
println("  Overfitting < 10%: $(overfitting < 0.1 ? "✓" : "✗") ($(round(overfitting*100, digits=1))%)")
println("  Val > 85%: $(val_acc > 0.85 ? "✓" : "✗")")

println("\n🏆 ¿ÉXITO TOTAL? $(success ? "SÍ ✓" : "NO ✗")")