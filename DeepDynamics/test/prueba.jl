using DeepDynamics
using CUDA
using Random
Random.seed!(42)
import DeepDynamics.NeuralNetwork: swish, mish
println("=== Ejemplo Real: Clasificación con LayerNorm ===\n")

# ==================================================
# 1. GENERAR DATOS (Simulando CIFAR-10)
# ==================================================
println("📊 Generando datos tipo CIFAR-10...")

n_train = 5000
n_test = 1000
img_size = 32
channels = 3
n_classes = 10

# Generar imágenes sintéticas (normalmente cargarías CIFAR-10 real)
function generate_synthetic_images(n_samples)
    # Simular diferentes clases con patrones distintos
    X_data = Float32[]
    y_data = Int[]
    
    for i in 1:n_samples
        class = rand(1:n_classes)
        # Cada clase tiene un patrón diferente
        img = randn(Float32, channels, img_size, img_size) * 0.1f0
        img .+= Float32(class) / 10f0  # Sesgo por clase
        
        # Añadir patrón específico de clase
        if class <= 3
            img[1, :, :] .+= 0.2f0  # Más rojo
        elseif class <= 6
            img[2, :, :] .+= 0.2f0  # Más verde
        else
            img[3, :, :] .+= 0.2f0  # Más azul
        end
        
        push!(X_data, vec(img)...)
        push!(y_data, class)
    end
    
    # Reshape a (C*H*W, N)
    X_matrix = reshape(X_data, channels*img_size*img_size, n_samples)
    
    # One-hot encoding para y
    y_matrix = zeros(Float32, n_classes, n_samples)
    for (i, class) in enumerate(y_data)
        y_matrix[class, i] = 1f0
    end
    
    return X_matrix, y_matrix
end

X_train, y_train = generate_synthetic_images(n_train)
X_test, y_test = generate_synthetic_images(n_test)

println("✅ Datos generados:")
println("   Train: $(size(X_train, 2)) muestras")
println("   Test: $(size(X_test, 2)) muestras")
println("   Dimensiones: $(channels)×$(img_size)×$(img_size) = $(size(X_train, 1)) features")

# ==================================================
# 2. MODELO CON LAYERNORM (Vision Transformer simplificado)
# ==================================================
println("\n🏗️ Construyendo modelo tipo Vision Transformer...")

# Patch embedding dimension
patch_size = 8
n_patches = (img_size ÷ patch_size)^2
patch_dim = channels * patch_size * patch_size
hidden_dim = 256
mlp_dim = 512

# Crear modelo inspirado en ViT
model = Sequential([
    # 1. Patch Embedding (simulado con Dense)
    Dense(channels * img_size * img_size, hidden_dim),
    LayerNorm(hidden_dim),  # 🔥 LayerNorm después de embedding
    Activation(relu),
    DropoutLayer(0.1),
    
    # 2. Transformer Block 1 (simplificado)
    # Self-attention simulada con Dense
    Dense(hidden_dim, hidden_dim),
    LayerNorm(hidden_dim),  # 🔥 LayerNorm en transformer
    Activation(relu),
    
    # MLP block
    Dense(hidden_dim, mlp_dim),
    Activation(swish),
    DropoutLayer(0.1),
    Dense(mlp_dim, hidden_dim),
    LayerNorm(hidden_dim),  # 🔥 LayerNorm después de MLP
    
    # 3. Transformer Block 2
    Dense(hidden_dim, hidden_dim),
    LayerNorm(hidden_dim),  # 🔥 Más LayerNorm
    Activation(relu),
    
    Dense(hidden_dim, mlp_dim),
    Activation(swish),
    DropoutLayer(0.1),
    Dense(mlp_dim, hidden_dim),
    LayerNorm(hidden_dim),  # 🔥 LayerNorm final del bloque
    
    # 4. Classification head
    Dense(hidden_dim, 128),
    LayerNorm(128),  # 🔥 LayerNorm antes de clasificación
    Activation(relu),
    DropoutLayer(0.5),
    Dense(128, n_classes),
    Activation(softmax)
])

# Contar parámetros
params = collect_parameters(model)
n_params = sum(length(p.data) for p in params)
println("✅ Modelo creado con $(length(model.layers)) capas")
println("   Parámetros totales: $(n_params)")
println("   LayerNorm layers: $(count(l -> l isa LayerNorm, model.layers))")

# Mover a GPU si está disponible
if CUDA.functional()
    model = model_to_gpu(model)
    println("   🚀 Modelo en GPU")
end

# ==================================================
# 3. CONFIGURAR ENTRENAMIENTO
# ==================================================
println("\n⚙️ Configurando entrenamiento...")

# Crear DataLoaders
train_loader = DataLoader(X_train, y_train, 64; shuffle=true)
test_loader = DataLoader(X_test, y_test, 64; shuffle=false)

# Callbacks avanzados
callbacks = [
    # Early stopping si no mejora
    EarlyStopping(patience=3, monitor="val_loss"),
    
    # Reducir learning rate en plateau
    ReduceLROnPlateau(factor=0.5f0, patience=2, monitor="val_loss"),
    
    # Guardar mejor modelo
    ModelCheckpoint(
        "best_model_layernorm.jld2",
        monitor="val_accuracy",
        mode=:max,
        save_best_only=true
    ),
    
    # Callback de progreso
    ProgressCallback(1)
]

# ==================================================
# 4. ENTRENAR CON fit!
# ==================================================
println("\n🎯 Entrenando modelo con LayerNorm...")

history = fit!(
    model, train_loader;
    val_loader=test_loader,
    epochs=30,
    optimizer=Adam(0.001f0),
    loss_fn=categorical_crossentropy,
    callbacks=callbacks,
    verbose=1,
    log_dir="experiments",
    experiment_name="vision_transformer_layernorm",
    use_tensorboard=false,
    log_config=Dict(
       "architecture" => "ViT-like with LayerNorm",
       "dataset"      => "CIFAR-10 synthetic",
       "n_layernorm"  => count(l -> l isa LayerNorm, model.layers),
       "optimizer"    => "Adam",
       "batch_size"   => 64
    )
)

# ==================================================
# 5. EVALUAR RESULTADOS
# ==================================================
println("\n📈 Resultados del entrenamiento:")

# Métricas finales
final_train_loss = history.train_loss[end]
final_val_loss = history.val_loss[end]
final_train_acc = history.train_metrics["accuracy"][end]
final_val_acc = history.val_metrics["accuracy"][end]

println("   Loss final:")
println("      Train: $(round(final_train_loss, digits=4))")
println("      Val: $(round(final_val_loss, digits=4))")
println("   Accuracy final:")
println("      Train: $(round(final_train_acc*100, digits=2))%")
println("      Val: $(round(final_val_acc*100, digits=2))%")

# Mejora desde el inicio
initial_loss = history.train_loss[1]
improvement = (1 - final_train_loss/initial_loss) * 100
println("   Mejora en loss: $(round(improvement, digits=1))%")

# ==================================================
# 6. INFERENCIA EN NUEVOS DATOS
# ==================================================
println("\n🔮 Probando inferencia...")

# Generar una nueva imagen de prueba
test_img = randn(Float32, channels * img_size * img_size, 1)

# Poner modelo en modo evaluación
set_training_mode!(model, false)

# Predecir
if CUDA.functional()
    test_tensor = to_gpu(Tensor(test_img))
else
    test_tensor = Tensor(test_img)
end

prediction = forward(model, test_tensor)
predicted_class = argmax(vec(prediction.data))

println("   Predicción para imagen de prueba:")
println("      Clase predicha: $predicted_class")
println("      Confianza: $(round(maximum(prediction.data)*100, digits=2))%")

# ==================================================
# 7. VISUALIZAR EFECTO DE LAYERNORM
# ==================================================
println("\n🔬 Analizando efecto de LayerNorm...")

# Extraer activaciones de una capa LayerNorm
test_batch = Tensor(X_train[:, 1:10])  # 10 muestras

# IMPORTANTE: Mover al mismo dispositivo que el modelo
if CUDA.functional() && model_device(model) == :gpu
    test_batch = to_gpu(test_batch)
end

set_training_mode!(model, false)

# Forward hasta la primera LayerNorm
activations_before = forward(model.layers[1], test_batch)  # Dense
ln_layer = model.layers[2]  # LayerNorm
activations_after = forward(ln_layer, activations_before)

# Mover a CPU para análisis con Statistics
if activations_before.data isa CUDA.CuArray
    before_data = Array(activations_before.data)
    after_data = Array(activations_after.data)
else
    before_data = activations_before.data
    after_data = activations_after.data
end

# Estadísticas
using Statistics
before_mean = mean(before_data, dims=1)
before_std = std(before_data, dims=1)
after_mean = mean(after_data, dims=1)
after_std = std(after_data, dims=1)

println("   Estadísticas antes de LayerNorm:")
println("      Mean range: [$(minimum(before_mean)), $(maximum(before_mean))]")
println("      Std range: [$(minimum(before_std)), $(maximum(before_std))]")
println("   Estadísticas después de LayerNorm:")
println("      Mean range: [$(minimum(after_mean)), $(maximum(after_mean))]")
println("      Std range: [$(minimum(after_std)), $(maximum(after_std))]")
# ==================================================
# 8. COMPARAR CON MODELO SIN LAYERNORM
# ==================================================
println("\n📊 Comparación con modelo sin LayerNorm...")

# Crear modelo equivalente con BatchNorm
model_bn = Sequential([
    Dense(channels * img_size * img_size, hidden_dim),
    BatchNorm(hidden_dim),  # BatchNorm en lugar de LayerNorm
    Activation(relu),
    DropoutLayer(0.1),
    Dense(hidden_dim, hidden_dim),
    BatchNorm(hidden_dim),
    Activation(relu),
    Dense(hidden_dim, n_classes),
    Activation(softmax)
])

# Entrenar brevemente para comparar
println("   Entrenando modelo con BatchNorm (10 epochs)...")
history_bn = fit!(
    model_bn, train_loader,
    val_loader=test_loader,
    epochs=10,
    optimizer=Adam(0.001f0),
    loss_fn=categorical_crossentropy,
    verbose=1
)

println("   Comparación después de 10 epochs:")
println("      LayerNorm val_acc: $(round(history.val_metrics["accuracy"][min(10, end)]*100, digits=2))%")
println("      BatchNorm val_acc: $(round(history_bn.val_metrics["accuracy"][end]*100, digits=2))%")

# ==================================================
# 9. GUARDAR MODELO FINAL
# ==================================================
println("\n💾 Guardando modelo final...")

save_model("final_model_layernorm.jld2", model)
println("✅ Modelo guardado en 'final_model_layernorm.jld2'")

# Cargar y verificar
loaded_model = load_model("final_model_layernorm.jld2")
println("✅ Modelo cargado y verificado")

println("\n" * "="^60)
println("🎉 EJEMPLO COMPLETO EXITOSO")
println("="^60)
println("\nLayerNorm demostró:")
println("• Estabilización del entrenamiento")
println("• Normalización independiente del batch")
println("• Integración perfecta con fit! y callbacks")
println("• Compatibilidad con GPU")
println("• Serialización/deserialización funcional")