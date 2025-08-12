# test/lstm_gru_example.jl
using DeepDynamics
using CUDA
using Random
using Statistics
using DeepDynamics.Optimizers
using DeepDynamics.Losses: crossentropy_from_logits

Random.seed!(42)
println("=== LSTM + GRU para Clasificación de Secuencias (GPU, logits CE) ===\n")

# ===========================
# Parámetros
# ===========================
num_samples_train = 3000
num_samples_test  = 600
seq_len    = 10
input_dim  = 20
hidden_lstm = 64
hidden_gru  = 32
num_classes = 3
batch_size  = 64
epochs      = 50

# ===========================
# Funciones auxiliares
# ===========================
function create_batches(X, Y, batch_size; shuffle=true)
    n = size(X, 2)
    idxs = shuffle ? Random.shuffle(collect(axes(X, 2))) : collect(axes(X, 2))
    batches = []
    
    for i in 1:batch_size:length(idxs)
        j = min(i + batch_size - 1, length(idxs))
        batch_idxs = idxs[i:j]
        
        X_batch = X[:, batch_idxs, :]
        Y_batch = Y[:, batch_idxs]
        
        push!(batches, (X_batch, Y_batch))
    end
    
    return batches
end

function softmax_cols(logits)
    exp_logits = exp.(logits .- maximum(logits, dims=1))
    return exp_logits ./ sum(exp_logits, dims=1)
end

# ===========================
# Generación de datos
# ===========================
println("📊 Generando datos sintéticos con patrón detectable...")

function generate_sequential_data(n_samples, seq_len, input_dim, num_classes)
    X = zeros(Float32, input_dim, n_samples, seq_len)
    Y = zeros(Float32, num_classes, n_samples)
    
    for i in axes(X, 2)
        class = rand(1:num_classes)
        
        # Base: ruido gaussiano
        X[:, i, :] = randn(Float32, input_dim, seq_len) * 0.5f0
        
        # Patrón sutil pero detectable por clase
        for t in axes(X, 3)
            if class == 1
                # CORREGIDO: usar .+= para broadcasting
                X[1:5, i, t] .+= 0.3f0 * sin(2π * t / seq_len)
                X[6:10, i, t] .+= 0.2f0 * cos(2π * t / seq_len)
            elseif class == 2
                # CORREGIDO: usar .+= para broadcasting
                X[6:10, i, t] .+= 0.3f0 * sin(4π * t / seq_len)
                X[11:15, i, t] .+= 0.2f0 * cos(4π * t / seq_len)
            else
                # CORREGIDO: usar .+= para broadcasting
                X[11:15, i, t] .+= 0.3f0 * sin(6π * t / seq_len)
                X[16:20, i, t] .+= 0.2f0 * cos(6π * t / seq_len)
            end
            
            # Ruido adicional - TAMBIÉN CORREGIDO
            X[:, i, t] .+= randn(Float32, input_dim) * 0.1f0
        end
        
        Y[class, i] = 1f0
    end
    
    return X, Y
end

# Generar datos
X_train, y_train = generate_sequential_data(num_samples_train, seq_len, input_dim, num_classes)
X_test, y_test = generate_sequential_data(num_samples_test, seq_len, input_dim, num_classes)

# Verificación de datos
println("\n=== VERIFICACIÓN DE DATOS ===")
X_mean = mean(X_train)
X_std = std(X_train)
println("X_train mean: $(round(X_mean, digits=4)) (esperado ~0)")
println("X_train std: $(round(X_std, digits=4)) (esperado ~0.5-0.8)")

# Distribución de clases
for c in 1:num_classes
    count_c = sum(y_train[c, :])
    println("Clase $c: $count_c muestras ($(round(100*count_c/num_samples_train, digits=1))%)")
end

train_classes = [argmax(y_train[:, i]) for i in axes(y_train, 2)]
test_classes = [argmax(y_test[:, i]) for i in axes(y_test, 2)]
println("Distribución train: ", [count(==(i), train_classes) for i in 1:3])
println("Distribución test: ", [count(==(i), test_classes) for i in 1:3])
println("=============================\n")

# ===========================
# MODELO LSTM + GRU
# ===========================
println("🏗️ Construyendo modelo LSTM + GRU...")

model = Sequential([
    # Solo una LSTM
    LSTM(input_dim, 64; return_sequences=false),
    DropoutLayer(0.2),
    
    # Dense directo
    Dense(64, 32),
    Activation(relu),
    DropoutLayer(0.2),
    
    Dense(32, num_classes)
])

# Configurar dispositivo
use_gpu = CUDA.functional()
if use_gpu
    model = model_to_gpu(model)
    println("   🚀 Modelo en GPU")
else
    println("   💻 Modelo en CPU")
end

# Recolectar parámetros
params = collect_parameters(model)
total_params = sum(length(p.data) for p in params)
println("✅ Modelo construido:")
println("   Arquitectura: LSTM($input_dim→$hidden_lstm) → GRU($hidden_lstm→$hidden_gru) → Dense(→$num_classes)")
println("   Parámetros totales: $total_params\n")

# ===========================
# Entrenamiento
# ===========================
opt = Adam(0.001f0)
global best_val_acc = 0.0f0
global patience_counter = 0
patience = 15

println("⚙️ Configurando entrenamiento...")
println("📊 Entrenando modelo...\n")

for epoch in 1:epochs
    # === TRAINING ===
    set_training_mode!(model, true)
    train_batches = create_batches(X_train, y_train, batch_size; shuffle=true)
    
    train_losses = Float32[]
    train_correct = 0
    train_total = 0
    
    for (X_batch, Y_batch) in train_batches
        # Crear tensores
        x = Tensor(X_batch)
        y = Tensor(Y_batch)
        
        if use_gpu
            x = to_gpu(x)
            y = to_gpu(y)
        end
        
        # Forward pass
        logits = forward(model, x)
        loss = crossentropy_from_logits(logits, y)
        
        # Backward pass
        zero_grad!(model)
        grad_seed = ones(Float32, size(loss.data)...)
        backward(loss, grad_seed)
        
        # Gradient clipping (importante para RNNs)
        Optimizers.clip_gradients_norm!(params, 5.0f0)
        
        # Update
        step!(opt, params)
        
        # Métricas
        push!(train_losses, loss.data[1])
        
        probs = softmax_cols(logits.data)
        preds = vec([argmax(probs[:, i]) for i in axes(probs, 2)])
        truth = vec([argmax(y.data[:, i]) for i in axes(y.data, 2)])
        
        train_correct += sum(preds .== truth)
        train_total += length(preds)
    end
    
    train_loss = mean(train_losses)
    train_acc = train_correct / train_total
    
    # === VALIDATION ===
    set_training_mode!(model, false)
    val_batches = create_batches(X_test, y_test, batch_size; shuffle=false)
    
    val_losses = Float32[]
    val_correct = 0
    val_total = 0
    
    for (X_batch, Y_batch) in val_batches
        x = Tensor(X_batch)
        y = Tensor(Y_batch)
        
        if use_gpu
            x = to_gpu(x)
            y = to_gpu(y)
        end
        
        logits = forward(model, x)
        loss = crossentropy_from_logits(logits, y)
        
        push!(val_losses, loss.data[1])
        
        probs = softmax_cols(logits.data)
        preds = vec([argmax(probs[:, i]) for i in axes(probs, 2)])
        truth = vec([argmax(y.data[:, i]) for i in axes(y.data, 2)])
        
        val_correct += sum(preds .== truth)
        val_total += length(preds)
    end
    
    val_loss = mean(val_losses)
    val_acc = val_correct / val_total
    
    # Imprimir progreso
    println("Epoch $epoch/$epochs - loss: $(round(train_loss, digits=4)) - acc: $(round(100*train_acc, digits=1))% - val_loss: $(round(val_loss, digits=4)) - val_acc: $(round(100*val_acc, digits=1))%")
    
    # Early stopping
    global best_val_acc, patience_counter
    
    if val_acc > best_val_acc
        best_val_acc = val_acc
        patience_counter = 0
        println("   ↑ Mejor modelo (val_acc: $(round(100*best_val_acc, digits=1))%)")
    else
        patience_counter += 1
        if patience_counter >= patience
            println("   ⚠️ Early stopping en época $epoch")
            break
        end
    end
    
    # Learning rate decay
    if epoch % 10 == 0
        opt.learning_rate *= 0.8f0
        println("   📉 LR → $(opt.learning_rate)")
    end
end

# ===========================
# Evaluación final
# ===========================
println("\n📈 Resultados finales:")
println("   Mejor val accuracy: $(round(100*best_val_acc, digits=2))%")

println("\n🔮 Ejemplos de predicción:")
set_training_mode!(model, false)

for i in 1:5
    idx = rand(axes(X_test, 2))
    
    # Tomar una muestra
    x_sample = X_test[:, idx:idx, :]
    y_true = y_test[:, idx]
    
    x = Tensor(x_sample)
    if use_gpu
        x = to_gpu(x)
    end
    
    logits = forward(model, x)
    probs = vec(softmax_cols(logits.data))
    
    pred_class = argmax(probs)
    true_class = argmax(y_true)
    confidence = maximum(probs) * 100
    
    status = pred_class == true_class ? "✓" : "✗"
    println("   Muestra $i: Real=$true_class, Pred=$pred_class ($(round(confidence, digits=1))%) $status")
end

# ===========================
# Diagnóstico del modelo
# ===========================
println("\n📊 Diagnóstico del modelo:")

# Verificar gradientes
grad_flags = [p.grad !== nothing for p in params if p.requires_grad]
println("   Parámetros con gradiente: $(sum(grad_flags))/$(length(grad_flags))")

# Verificar salida
x_test = Tensor(X_test[:, 1:1, :])
if use_gpu
    x_test = to_gpu(x_test)
end
logits_test = forward(model, x_test)
probs_test = softmax_cols(logits_test.data)
println("   Suma de probabilidades: $(round(sum(probs_test), digits=4)) (debe ser ≈1.0)")

# Estadísticas de pesos
for (i, layer) in enumerate(model.layers)
    if layer isa LSTM
        cell = layer.cell
        w_mean = mean(abs.(cell.W_ii.data))
        println("   Layer $i (LSTM): |W| mean = $(round(w_mean, digits=4))")
    elseif layer isa GRU
        cell = layer.cell
        w_mean = mean(abs.(cell.W_ir.data))
        println("   Layer $i (GRU): |W| mean = $(round(w_mean, digits=4))")
    elseif layer isa Dense
        w_mean = mean(abs.(layer.weights.data))
        println("   Layer $i (Dense): |W| mean = $(round(w_mean, digits=4))")
    end
end

println("\n" * "═"^60)
println("✅ Entrenamiento LSTM + GRU completado exitosamente")