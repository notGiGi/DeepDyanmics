using DeepDynamics
using Random
using Statistics: mean, std, var
using LinearAlgebra: norm
using CUDA
using DeepDynamics.GPUMemoryManager
using DeepDynamics.Training: set_training_mode!
using DeepDynamics.TensorEngine: zero_grad!, backward
using DeepDynamics.NeuralNetwork: forward, collect_parameters, relu, softmax
using DeepDynamics.Losses: categorical_crossentropy
using DeepDynamics.Optimizers: Adam
using DeepDynamics: model_to_gpu, to_gpu, Tensor, optim_step!, Dense, Sequential, BatchNorm, Activation, DropoutLayer
Random.seed!(42)

# ===============================================
# EJEMPLO COMPLETO FASE 8: Red Neuronal Robusta
# ===============================================

println("🚀 DeepDynamics - Ejemplo Fase 8: Mejoras de Estabilidad")
println("="^60)

# 1. GENERAR DATOS SINTÉTICOS MÁS RETADORES
# Problema: Clasificación con patrones complejos pero aprendibles
function generate_challenging_data(n_samples=1000, n_features=12, n_classes=3)
    n_samples -= n_samples % n_classes  # ⚠️ Asegura divisibilidad
    samples_per_class = n_samples ÷ n_classes
    X = Float32[]
    y = Float32[]

    for class in 1:n_classes
        for _ in 1:samples_per_class
            x = zeros(Float32, n_features)
            
            # 1. Mezcla de distribuciones
            if class == 1
                x[1:4] .= randn(Float32, 4) .* 1.5 .+ 2.0              # Gaussiana desplazada
                x[5:8] .= rand(Float32, 4) .* 3.0                       # Uniforme
                x[9] = sin(x[2]) + 0.1f0 * randn(Float32)              # Dependencia no lineal
                x[10] = x[1] * x[3] + randn(Float32) * 0.3f0            # Correlación
                x[11] = exp(-abs(x[4])) + randn(Float32) * 0.05f0
                x[12] = -x[6]^2 + randn(Float32) * 0.2f0

            elseif class == 2
                x[1:6] .= randn(Float32, 6) .* 0.8                     # Más concentrada
                x[7:9] .= rand(Float32, 3) .* 5.0
                x[10] = log(abs(x[2]) + 1f-2) + 0.2f0 * randn()
                x[11] = x[5] - x[6]^2 + 0.1f0 * randn()
                x[12] = sin(x[8] + x[9]) + 0.1f0 * randn()

            else
                x[1:3] .= randn(Float32, 3) .* 2.0
                x[4:6] .= rand(Float32, 3) .* 2.0
                x[7:9] .= randn(Float32, 3) .* 0.5
                x[10] = x[1]*x[2]*x[3] + randn(Float32) * 0.5
                x[11] = cos(x[4] + x[5]) + 0.1f0 * randn()
                x[12] = x[6] * x[9] + 0.1f0 * randn()
            end

            append!(X, x)

            label = zeros(Float32, n_classes)
            label[class] = 1.0f0
            append!(y, label)
        end
    end

    # Mezclar y normalizar
    indices = shuffle(1:n_samples)
    X_data = reshape(X, n_features, n_samples)[:, indices]
    y_data = reshape(y, n_classes, n_samples)[:, indices]

    X_mean = mean(X_data, dims=2)
    X_std = std(X_data, dims=2) .+ 1f-6
    X_data = (X_data .- X_mean) ./ X_std

    return X_data, y_data
end


# Generar datos
println("\n📊 Generando datos sintéticos complejos...")
X_train, y_train = generate_challenging_data(800)
X_val, y_val = generate_challenging_data(200)


println("  Dimensiones X_train: ", size(X_train))
println("  Dimensiones y_train: ", size(y_train))
println("  Clases balanceadas: ", sum(y_train, dims=2))
println("  Rango X_train: [$(minimum(X_train)), $(maximum(X_train))]")

# Convertir a Tensors
X_train = Tensor(X_train; requires_grad=false)
y_train = Tensor(y_train; requires_grad=false)
X_val = Tensor(X_val; requires_grad=false)
y_val = Tensor(y_val; requires_grad=false)

# 2. DEFINIR MODELO OPTIMIZADO CON FASE 8
println("\n🏗️ Construyendo modelo con mejoras Fase 8...")
model = Sequential([
    # Entrada: 12 features
    Dense(12, 32),
    BatchNorm(32),
    Activation(relu),

    Dense(32, 64),
    BatchNorm(64),
    Activation(relu),
    DropoutLayer(0.2, true),

    Dense(64, 32),
    BatchNorm(32),
    Activation(relu),
    DropoutLayer(0.2, true),

    Dense(32, 16),
    BatchNorm(16),
    Activation(relu),

    Dense(16, 3),
    Activation(softmax)
])



println("  Modelo creado con $(length(model.layers)) capas")

# 3. CONFIGURAR OPTIMIZADOR
# Learning rate más alto para convergencia más rápida
opt = Adam(learning_rate=0.01, weight_decay=0.0001)

# Mover a GPU si está disponible
if CUDA.functional()
    println("  🖥️ GPU detectada, moviendo modelo...")
    model = model_to_gpu(model)
    X_train = to_gpu(X_train)
    y_train = to_gpu(y_train)
    X_val = to_gpu(X_val)
    y_val = to_gpu(y_val)
end

# 4. FUNCIÓN DE EVALUACIÓN MEJORADA
function evaluate_model(model, X, y; mode=:eval)
    # Cambiar a modo evaluación
    set_training_mode!(model, mode == :train)
    
    # Forward pass
    pred = forward(model, X)
    
    # Calcular accuracy
    pred_cpu = pred.data isa CUDA.CuArray ? Array(pred.data) : pred.data
    y_cpu = y.data isa CUDA.CuArray ? Array(y.data) : y.data
    
    pred_classes = [argmax(pred_cpu[:, i]) for i in axes(pred_cpu, 2)]

    true_classes = [argmax(y_cpu[:, i]) for i in axes(y_cpu, 2)]

    accuracy = mean(pred_classes .== true_classes) * 100
    
    # Calcular loss
    loss = categorical_crossentropy(pred, y)
    
    return loss.data[1], accuracy
end

# 5. ENTRENAMIENTO CON MONITOREO DETALLADO
println("\n🎯 Iniciando entrenamiento...")
println("="^60)

n_epochs = 30
batch_size = 32
best_val_acc = 0.0
patience = 5
no_improve_count = 0

# Guardar historial
train_losses = Float32[]
val_losses = Float32[]
train_accs = Float32[]
val_accs = Float32[]

for epoch in 1:n_epochs
    global best_val_acc, no_improve_count
    
    # MODO TRAINING
    set_training_mode!(model, true)
    
    # Entrenar una época manualmente
    params = collect_parameters(model)
    n_batches = div(size(X_train.data, 2), batch_size)
    
    for batch_idx in 1:n_batches
        try
            # Zero gradients
            for p in params
                zero_grad!(p)
            end
            
            # Get batch
            start_idx = (batch_idx - 1) * batch_size + 1
            end_idx = min(batch_idx * batch_size, size(X_train.data, 2))
            
            # Create batch tensors en el mismo dispositivo
            batch_data_X = X_train.data[:, start_idx:end_idx]
            batch_data_y = y_train.data[:, start_idx:end_idx]
            
            if CUDA.functional() && batch_data_X isa CUDA.CuArray
                X_batch = Tensor(batch_data_X; requires_grad=false)
                y_batch = Tensor(batch_data_y; requires_grad=false)
            else
                X_batch = Tensor(batch_data_X; requires_grad=false)
                y_batch = Tensor(batch_data_y; requires_grad=false)
            end
            
            # Forward pass
            output = forward(model, X_batch)
            loss = categorical_crossentropy(output, y_batch)
            
            # Backward pass
            if loss.data isa CUDA.CuArray
                backward(loss, CUDA.CuArray([1.0f0]))
            else
                backward(loss, [1.0f0])
            end
            
            # Update weights
            optim_step!(opt, params)
        catch e
            if batch_idx == 1
                println("  ⚠️ Error en batch $batch_idx: ", e)
                rethrow(e)
            end
        end
    end
    
    # MODO EVALUACIÓN
    train_loss, train_acc = evaluate_model(model, X_train, y_train; mode=:eval)
    val_loss, val_acc = evaluate_model(model, X_val, y_val; mode=:eval)
    
    # Guardar métricas
    push!(train_losses, train_loss)
    push!(val_losses, val_loss)
    push!(train_accs, train_acc)
    push!(val_accs, val_acc)
    
    # Early stopping
    if val_acc > best_val_acc
        best_val_acc = val_acc
        no_improve_count = 0
        println("  🌟 Nueva mejor validación en época")
    else
        no_improve_count += 1
    end
    
    # Imprimir progreso
    if epoch % 3 == 0 || epoch == 1 || (val_acc > best_val_acc - 5.0 && epoch > 5)
        println("Época $epoch/$n_epochs:")
        println("  📈 Loss - Train: $(round(train_loss, digits=4)), Val: $(round(val_loss, digits=4))")
        println("  🎯 Acc  - Train: $(round(train_acc, digits=2))%, Val: $(round(val_acc, digits=2))%")
        
        # Mostrar mejora desde el inicio
        if epoch > 1 && length(train_accs) > 0
            initial_acc = train_accs[1]
            improvement = train_acc - initial_acc
            println("  📊 Mejora desde inicio: $(improvement >= 0 ? "+" : "")$(round(improvement, digits=2))%")
        end
    end
    
    # Early stopping
    if no_improve_count >= patience && epoch > 10
        println("\n⏹️ Early stopping en época $epoch")
        break
    end
end

# Asegurar que el modelo esté en modo eval al final
set_training_mode!(model, false)

println("\n✅ Entrenamiento completado!")
println("  Accuracy inicial: $(round(train_accs[1], digits=2))%")
println("  Accuracy final: $(round(train_accs[end], digits=2))%")
println("  Mejor validation accuracy: $(round(best_val_acc, digits=2))%")
println("  Mejora total: +$(round(train_accs[end] - train_accs[1], digits=2))%")

# 6. ANÁLISIS DE COMPORTAMIENTO
println("\n📊 Análisis de convergencia:")
if length(train_losses) > 5
    # Verificar tendencia de loss
    recent_losses = train_losses[end-4:end]
    loss_trend = recent_losses[1] - recent_losses[end]
    println("  Reducción de loss (últimas 5 épocas): $(round(loss_trend, digits=4))")
    
    # Verificar estabilidad
    loss_std = std(recent_losses)
    println("  Desviación estándar del loss (estabilidad): $(round(loss_std, digits=6))")
end

# 7. ANÁLISIS DE CAPAS BATCHNORM
println("\n🔍 Análisis de estabilidad del modelo:")
bn_layers = [layer for layer in model.layers if layer isa BatchNorm]
println("  BatchNorm layers: $(length(bn_layers))")
for (i, bn) in enumerate(bn_layers)
    println("    BN$i - tracked batches: $(bn.num_batches_tracked)")
    mean_range = (minimum(bn.running_mean), maximum(bn.running_mean))
    var_range = (minimum(bn.running_var), maximum(bn.running_var))
    println("    BN$i - running mean range: [$(round(mean_range[1], digits=4)), $(round(mean_range[2], digits=4))]")
    println("    BN$i - running var range: [$(round(var_range[1], digits=4)), $(round(var_range[2], digits=4))]")
end

# 8. TEST DE ROBUSTEZ
println("\n🛡️ Tests de robustez:")

# Test 1: Entrada con ruido
X_noisy = Tensor(randn(Float32, 12, 5) * 3.0f0; requires_grad=false)

if CUDA.functional()
    X_noisy = to_gpu(X_noisy)
end
set_training_mode!(model, false)
pred_noisy = forward(model, X_noisy)
pred_noisy_cpu = pred_noisy.data isa CUDA.CuArray ? Array(pred_noisy.data) : pred_noisy.data
println("  ✅ Entrada ruidosa - Salida estable: $(all(0 .<= pred_noisy_cpu .<= 1))")
sums = [sum(pred_noisy_cpu[:, i]) for i in axes(pred_noisy_cpu, 2)]

println("  Suma de probabilidades: $(round.(sums, digits=4))")

# Test 2: Consistencia en modo eval
test_indices = 1:min(5, size(X_val.data, 2))
test_data = X_val.data[:, test_indices]
test_input = Tensor(test_data; requires_grad=false)
out1 = forward(model, test_input)
out2 = forward(model, test_input)
out1_cpu = out1.data isa CUDA.CuArray ? Array(out1.data) : out1.data
out2_cpu = out2.data isa CUDA.CuArray ? Array(out2.data) : out2.data
println("  ✅ Consistencia en eval: $(all(out1_cpu .≈ out2_cpu))")

# 9. DEMOSTRACIÓN DE DIFERENCIAS TRAINING/EVAL
println("\n🔄 Comportamiento Training vs Eval:")
sample_data = X_val.data[:, 1:1]
sample = Tensor(sample_data; requires_grad=false)

set_training_mode!(model, true)
train_outputs = []
for _ in 1:5
    out = forward(model, sample)
    out_cpu = out.data isa CUDA.CuArray ? Array(out.data) : out.data
    push!(train_outputs, out_cpu)
end
train_std = std([out[1] for out in train_outputs])

set_training_mode!(model, false)
eval_outputs = []
for _ in 1:5
    out = forward(model, sample)
    out_cpu = out.data isa CUDA.CuArray ? Array(out.data) : out.data
    push!(eval_outputs, out_cpu)
end
eval_std = std([out[1] for out in eval_outputs])

println("  Variabilidad en training: $(round(train_std, digits=6))")
println("  Variabilidad en eval: $(round(eval_std, digits=6))")
println("  Dropout activo: $(train_std > eval_std * 10)")

# 10. LIMPIEZA GPU
if CUDA.functional()
    println("\n🧹 Gestión de memoria GPU:")
    stats = GPUMemoryManager.memory_stats()
    println("  Memoria total: $(round(stats.total, digits=2)) GB")
    println("  Memoria usada: $(round(stats.used, digits=2)) GB ($(round(stats.free_percent, digits=1))% libre)")

    
    GPUMemoryManager.clear_cache()
    println("  ✅ Cache limpiado")
end

println("\n🎉 Ejemplo Fase 8 completado exitosamente!")
println("="^60)