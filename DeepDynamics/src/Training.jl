module Training

using ..TensorEngine
using ..NeuralNetwork
using ..Optimizers
using ..Visualizations
using ..GPUMemoryManager  # Nuevo import para gestión de memoria
using ..DataLoaders       # Nuevo import para DataLoaders optimizados
using Random
using LinearAlgebra
using CUDA
import ..Optimizers: step!
export train!, train_batch!, compute_accuracy_general, train_improved!,
       EarlyStopping, Callback, PrintCallback, FinalReportCallback, add_callback!, 
       run_epoch_callbacks, run_final_callbacks, train_with_loaders, stack_batch, evaluate_model

# -----------------------------------------------------------------------
# Estructuras de EarlyStopping, Callbacks, etc.
# -----------------------------------------------------------------------

mutable struct EarlyStopping
    patience::Int
    min_delta::Float64
    best_loss::Float64
    wait::Int
    stopped::Bool
    function EarlyStopping(; patience::Int = 5, min_delta::Float64 = 0.0)
        new(patience, min_delta, Inf, 0, false)
    end
end

function should_stop_early!(es::EarlyStopping, val_loss::Float64)
    if val_loss < es.best_loss - es.min_delta
        es.best_loss = val_loss
        es.wait = 0
    else
        es.wait += 1
    end
    if es.wait ≥ es.patience
        es.stopped = true
    end
end

abstract type Callback end

struct PrintCallback <: Callback
    freq::Int
end

function on_epoch_end(cb::PrintCallback, epoch::Int, train_loss, val_loss, train_acc, val_acc)
    println("Epoch $epoch: Training Loss = $(round(train_loss, digits=4)), Validation Loss = $(round(val_loss, digits=4)), Training Accuracy = $(round(train_acc*100, digits=2))%, Validation Accuracy = $(round(val_acc*100, digits=2))%")
end

struct FinalReportCallback <: Callback
    output_format::Symbol
    filename::String
end

function on_training_end(cb::FinalReportCallback, train_losses::Vector{Float64}, val_losses::Vector{Float64})
    Reports.generate_report(train_losses, val_losses; output_format=cb.output_format, filename=cb.filename)
    println("Final report generated and saved to ", cb.filename)
end

const callbacks = Callback[]

function add_callback!(cb::Callback)
    push!(callbacks, cb)
end

function run_epoch_callbacks(epoch::Int, train_loss, val_loss, train_acc, val_acc)
    for cb in callbacks
        if cb isa PrintCallback
            on_epoch_end(cb, epoch, train_loss, val_loss, train_acc, val_acc)
        end
    end
end

function run_final_callbacks(train_losses::Vector{Float64}, val_losses::Vector{Float64})
    for cb in callbacks
        if cb isa FinalReportCallback
            on_training_end(cb, train_losses, val_losses)
        end
    end
end

# -----------------------------------------------------------------------
# Función de stacking optimizada para GPU
# -----------------------------------------------------------------------
# Mejora para stack_batch en Training.jl
# Reemplazar la función stack_batch existente con esta versión mejorada

function stack_batch(batch::Vector{<:TensorEngine.Tensor})
    isempty(batch) && return TensorEngine.Tensor(zeros(Float32, 0))
    
    # Detectar dispositivo del primer tensor
    first_tensor = batch[1]
    target_device = first_tensor.data isa CUDA.CuArray ? :gpu : :cpu
    
    # Verificar que todos los tensores estén en el mismo dispositivo
    # Si no, moverlos al dispositivo del primer tensor
    consistent_batch = Vector{TensorEngine.Tensor}(undef, length(batch))
    for (i, tensor) in enumerate(batch)
        current_device = tensor.data isa CUDA.CuArray ? :gpu : :cpu
        if current_device != target_device
            if target_device == :gpu
                # Mover a GPU
                new_data = CUDA.CuArray(tensor.data)
                consistent_batch[i] = TensorEngine.Tensor(new_data; requires_grad=tensor.requires_grad)
            else
                # Mover a CPU
                new_data = Array(tensor.data)
                consistent_batch[i] = TensorEngine.Tensor(new_data; requires_grad=tensor.requires_grad)
            end
        else
            consistent_batch[i] = tensor
        end
    end
    
    # Ahora procesar según dimensiones
    nd = TensorEngine.ndims(first_tensor)
    is_on_gpu = (target_device == :gpu)
    
    if nd == 3  # Tensores 3D (C, H, W)
        if is_on_gpu
            # Versión optimizada para GPU
            c, h, w = size(first_tensor.data)
            n = length(consistent_batch)
            
            # Usar GPUMemoryManager para obtener buffer eficiente
            result_buffer = GPUMemoryManager.get_tensor_buffer((n, c, h, w), Float32)
            
            # Copiar datos
            for (i, tensor) in enumerate(consistent_batch)
                result_buffer[i, :, :, :] = tensor.data
            end
            
            return TensorEngine.Tensor(result_buffer)
        else
            # Versión CPU
            stacked = cat([t.data for t in consistent_batch]..., dims=4)  # (C, H, W, N)
            stacked = permutedims(stacked, (4, 1, 2, 3))               # (N, C, H, W)
            return TensorEngine.Tensor(stacked)
        end
        
    elseif nd == 4  #Tensores 4D
        # Determinar formato inspeccionando dimensiones
        first_dims = size(first_tensor.data)
        
        # Heurística mejorada para detectar formato
        is_nchw = detect_format(first_dims) == :NCHW
        
        if is_nchw
            # Para formato NCHW, concatenar a lo largo de la primera dimensión (batch)
            if is_on_gpu
                # GPU: usar cat optimizado
                stacked_data = cat([t.data for t in consistent_batch]..., dims=1)
            else
                # CPU: usar vcat para eficiencia
                stacked_data = vcat([t.data for t in consistent_batch]...)
            end
        else
            # Formato WHCN: concatenar en la última dimensión
            stacked_data = cat([t.data for t in consistent_batch]..., dims=4)
        end
        
        return TensorEngine.Tensor(stacked_data)
        
    elseif nd == 1 || nd == 2  # Etiquetas
        if is_on_gpu
            # GPU: concatenar en dimensión 2 para mantener formato (features, batch)
            return TensorEngine.Tensor(cat([t.data for t in consistent_batch]..., dims=2))
        else
            # CPU: usar hcat
            return TensorEngine.Tensor(hcat([t.data for t in consistent_batch]...))
        end
        
    else
        error("Formato no soportado: ndims=$nd")
    end
end

# Función auxiliar para detectar formato (reutilizar de Flatten)
function detect_format(dims::Tuple)
    if length(dims) != 4
        return :UNKNOWN
    end
    
    # Heurísticas para detectar formato:
    # NCHW: batch suele ser pequeño (1-256), canales moderados (3-2048)
    # WHCN: batch al final, dimensiones espaciales primero
    
    # Si la primera dimensión es pequeña y la segunda parece canales
    if dims[1] <= 256 && dims[2] in [1, 3, 16, 32, 64, 128, 256, 512, 1024, 2048]
        return :NCHW
    # Si las primeras dos dimensiones son grandes (espaciales) y la última es pequeña (batch)
    elseif dims[1] > 10 && dims[2] > 10 && dims[4] <= 256
        return :WHCN
    else
        # Default a NCHW si no está claro
        return :NCHW
    end
end

# -----------------------------------------------------------------------
# Cálculo de accuracy optimizado
# -----------------------------------------------------------------------
function compute_accuracy_general(model, inputs::Vector{<:TensorEngine.Tensor}, targets::Vector{<:TensorEngine.Tensor})
    # Para conjuntos grandes, usar un enfoque por lotes para eficiencia
    if length(inputs) > 100 && CUDA.functional()
        return compute_accuracy_batched(model, inputs, targets)
    end
    
    correct = 0
    for i in 1:length(inputs)
        # Adaptar la imagen a 4D si es necesario
        input = inputs[i]
        if ndims(input.data) == 3
            input = adapt_image(input)
        end
        
        # Mover a GPU si es posible para mejor rendimiento
        if CUDA.functional() && !(input.data isa CUDA.CuArray)
            input = TensorEngine.to_gpu(input)
        end
        
        output = model(input)
        pred = argmax(vec(output.data))
        
        # Manejar tanto one-hot como etiquetas escalares
        true_label = length(targets[i].data) > 1 ? argmax(vec(targets[i].data)) : round(Int, targets[i].data[1])
        
        correct += (pred == true_label)
    end
    return correct / length(inputs)
end

# Versión por lotes para mayor eficiencia
function compute_accuracy_batched(model, inputs::Vector{<:TensorEngine.Tensor}, targets::Vector{<:TensorEngine.Tensor}; batch_size=32)
    n = length(inputs)
    correct = 0
    
    for start_idx in 1:batch_size:n
        end_idx = min(start_idx + batch_size - 1, n)
        batch_inputs = inputs[start_idx:end_idx]
        batch_targets = targets[start_idx:end_idx]
        
        # Adaptar imágenes si es necesario
        batch_inputs = [
            ndims(img.data) == 3 ? adapt_image(img) : img
            for img in batch_inputs
        ]
        
        # Mover a GPU si es posible
        if CUDA.functional()
            batch_inputs = [TensorEngine.to_gpu(img) for img in batch_inputs]
            batch_targets = [TensorEngine.to_gpu(tgt) for tgt in batch_targets]
        end
        
        # Apilar en batches
        stacked_inputs = stack_batch(batch_inputs)
        
        # Forward pass
        outputs = model(stacked_inputs)
        
        # Calcular precisión
        for i in 1:length(batch_inputs)
            pred_idx = argmax(outputs.data[:, i])
            true_idx = argmax(batch_targets[i].data)
            correct += (pred_idx == true_idx)
        end
    end
    
    return correct / n
end

# -----------------------------------------------------------------------
# Función interna para procesar un batch (optimizada para GPU)
# -----------------------------------------------------------------------
function _process_batch(batch_idxs, train_inputs, train_targets, model, loss_fn)
    batch_imgs = [train_inputs[i] for i in batch_idxs]
    batch_labels = [train_targets[i] for i in batch_idxs]
    
    # Adaptar imágenes si es necesario
    batch_imgs = [
        ndims(img.data) == 3 ? adapt_image(img) : img
        for img in batch_imgs
    ]
    
    # Mover a GPU si es posible
    if CUDA.functional()
        batch_imgs = [TensorEngine.to_gpu(img) for img in batch_imgs]
        batch_labels = [TensorEngine.to_gpu(lbl) for lbl in batch_labels]
    end
    
    # Stack inputs y labels
    input_batch = stack_batch(batch_imgs)
    label_batch = stack_batch(batch_labels)
    
    # Forward pass
    output = NeuralNetwork.forward(model, input_batch)
    
    # Calcular pérdida
    loss = loss_fn(output, label_batch)
    
    # Backpropagation
    if loss.backward_fn !== nothing
        loss.backward_fn(ones(size(loss.data)))
    end
    
    return loss.data[1]
end

# -----------------------------------------------------------------------
# Funciones de entrenamiento optimizadas para GPU
# -----------------------------------------------------------------------
function train_batch!(model::NeuralNetwork.Sequential, optimizer, loss_fn::Function,
                      train_inputs::Vector{<:TensorEngine.Tensor},
                      train_targets::Vector{<:TensorEngine.Tensor},
                      epochs::Int;
                      batch_size::Int=32, verbose::Bool=false, initial_lr::Float64=0.01,
                      decay_rate::Float64=0.99,
                      val_inputs::Vector{<:TensorEngine.Tensor}=TensorEngine.Tensor[],
                      val_targets::Vector{<:TensorEngine.Tensor}=TensorEngine.Tensor[],
                      early_stopping::Union{Nothing, EarlyStopping}=nothing,
                      visualize::Bool=false, profile::Bool=false)

    empty!(callbacks)
    if verbose && isempty(filter(x->x isa PrintCallback, callbacks))
        add_callback!(PrintCallback(1))
    end
    
    train_losses = Float64[]
    val_losses   = Float64[]
    params = NeuralNetwork.collect_parameters(model)
    num_samples = length(train_inputs)

    for epoch in 1:epochs
        # Update LR si corresponde
        if hasproperty(optimizer, :learning_rate)
            optimizer.learning_rate = initial_lr * decay_rate^(epoch-1)
        end
        
        epoch_loss = 0.0
        idxs = shuffle(1:num_samples)
        num_batches = ceil(Int, num_samples / batch_size)

        # Bucle de batches
        for bstart in 1:batch_size:num_samples
            bend = min(bstart+batch_size-1, num_samples)
            batch_idxs = idxs[bstart:bend]
            
            # CAMBIO FASE 1: Usar zero_grad! en lugar de initialize_grad!
            for p in params
                TensorEngine.zero_grad!(p)
            end

            # Procesar batch
            batch_loss = _process_batch(batch_idxs, train_inputs, train_targets, model, loss_fn)
            # Actualizar parámetros
            step!(optimizer, params)
            epoch_loss += batch_loss
            
            # Limpiar memoria GPU después de cada batch si estamos usando CUDA
            if CUDA.functional()
                GC.gc()
                CUDA.reclaim()
            end
        end

        epoch_loss /= num_batches
        push!(train_losses, epoch_loss)

        if !isempty(val_inputs)
            # Calcular pérdida de validación por lotes
            val_loss_sum = 0.0
            val_batches = 0
            
            for vstart in 1:batch_size:length(val_inputs)
                vend = min(vstart+batch_size-1, length(val_inputs))
                val_idxs = vstart:vend
                
                val_batch_imgs = [val_inputs[i] for i in val_idxs]
                val_batch_labels = [val_targets[i] for i in val_idxs]
                
                # Adaptar imágenes si es necesario
                val_batch_imgs = [
                    ndims(img.data) == 3 ? adapt_image(img) : img
                    for img in val_batch_imgs
                ]
                
                # Mover a GPU si es posible
                if CUDA.functional()
                    val_batch_imgs = [TensorEngine.to_gpu(img) for img in val_batch_imgs]
                    val_batch_labels = [TensorEngine.to_gpu(lbl) for lbl in val_batch_labels]
                end
                
                val_input_batch = stack_batch(val_batch_imgs)
                val_label_batch = stack_batch(val_batch_labels)
                
                val_out = NeuralNetwork.forward(model, val_input_batch)
                val_batch_loss = loss_fn(val_out, val_label_batch)
                
                val_loss_sum += val_batch_loss.data[1]
                val_batches += 1
            end
            
            val_loss = val_loss_sum / val_batches
            push!(val_losses, val_loss)
        else
            push!(val_losses, NaN)
        end
        
        # Calcular accuracies para callbacks
        train_acc = compute_accuracy_general(model, train_inputs, train_targets)
        val_acc   = compute_accuracy_general(model, val_inputs, val_targets)
        
        run_epoch_callbacks(epoch, train_losses[end], val_losses[end], train_acc, val_acc)
        
        if early_stopping !== nothing && !isempty(val_inputs)
            should_stop_early!(early_stopping, val_losses[end])
            if early_stopping.stopped
                println("Early stopping triggered at epoch $epoch.")
                run_final_callbacks(train_losses, val_losses)
                return train_losses, val_losses, epoch
            end
        end
        
        # Limpiar caché de memoria GPU al final de cada época
        if CUDA.functional()
            GPUMemoryManager.clear_cache()
            GC.gc()
            CUDA.reclaim()
        end
    end

    if visualize
        Visualizations.plot_metrics(train_losses, val_losses)
    end
    run_final_callbacks(train_losses, val_losses)
    return train_losses, val_losses, epochs
end

# -----------------------------------------------------------------------
# Alias para train!
# -----------------------------------------------------------------------
train!(args...; kwargs...) = train_batch!(args...; kwargs...)

# -----------------------------------------------------------------------
# DataLoader y funciones relacionadas
# -----------------------------------------------------------------------

"""
    train_with_loaders(model, optimizer, loss_fn, train_loader, val_loader, epochs, batch_size)

Versión optimizada de entrenamiento que usa DataLoaders para mejor rendimiento en GPU.
"""
function train_with_loaders(model, optimizer, loss_fn, train_loader, val_loader, epochs, batch_size)
    train_losses = Float64[]
    val_losses = Float64[]
    train_accs = Float64[]
    val_accs = Float64[]
    
    params = NeuralNetwork.collect_parameters(model)
    
    # Variables para early stopping
    best_val_loss = Inf
    wait_epochs = 0
    patience = 10
    min_delta = 0.0001
    best_weights = nothing
    
    for epoch in 1:epochs
        println("Epoch $epoch/$epochs:")
        epoch_start = time()
        
        # Training
        epoch_loss = 0.0
        num_batches = 0
        
        for (batch_x, batch_y) in train_loader
            # CAMBIO FASE 1: Usar zero_grad!
            for param in params
                TensorEngine.zero_grad!(param)
            end
            
            # Forward pass
            output = NeuralNetwork.forward(model, batch_x)
            
            # Calcular loss
            loss = loss_fn(output, batch_y)
            
            # Backpropagation
            if loss.backward_fn !== nothing
                loss.backward_fn(ones(size(loss.data)))
            end
            
            # Actualizar parámetros
            step!(optimizer, params)
            
            # Acumular pérdida
            epoch_loss += loss.data[1]
            num_batches += 1
            
            # Liberar memoria GPU después de cada batch
            if CUDA.functional()
                GC.gc()
                CUDA.reclaim()
            end
        end
        
        # Calcular pérdida promedio de la época
        epoch_loss /= num_batches
        push!(train_losses, epoch_loss)
        
        # Evaluar en validación
        val_loss, val_acc = evaluate_model(model, loss_fn, val_loader)
        push!(val_losses, val_loss)
        
        # Calcular exactitud en entrenamiento
        train_acc = compute_accuracy_with_loader(model, train_loader)
        push!(train_accs, train_acc)
        push!(val_accs, val_acc)
        
        # Imprimir resultados
        epoch_time = time() - epoch_start
        println("  Pérdida: train=$(round(epoch_loss, digits=4)), val=$(round(val_loss, digits=4))")
        println("  Exactitud: train=$(round(train_acc*100, digits=2))%, val=$(round(val_acc*100, digits=2))%")
        println("  Tiempo: $(round(epoch_time, digits=2))s")
        
        # Early stopping check
        if val_loss < best_val_loss - min_delta
            best_val_loss = val_loss
            wait_epochs = 0
            best_weights = deepcopy([param.data for param in params])
        else
            wait_epochs += 1
            if wait_epochs >= patience
                println("Early stopping triggered")
                # Restaurar mejores pesos
                if best_weights !== nothing
                    for (param, weights) in zip(params, best_weights)
                        param.data = copy(weights)
                    end
                end
                break
            end
        end
        
        # Limpiar caché de memoria GPU al final de cada época
        if CUDA.functional()
            GPUMemoryManager.clear_cache()
            GC.gc()
            CUDA.reclaim()
        end
    end
    
    return train_losses, val_losses, train_accs, val_accs
end

"""
    compute_accuracy_with_loader(model, data_loader)

Calcula la exactitud usando un DataLoader.
"""
function compute_accuracy_with_loader(model, data_loader)
    correct = 0
    total = 0
    
    for (batch_x, batch_y) in data_loader
        output = NeuralNetwork.forward(model, batch_x)
        
        # Obtener predicciones (máximo por columna)
        predictions = argmax(output.data, dims=1)
        labels = argmax(batch_y.data, dims=1)
        
        # Contar aciertos
        correct += sum(predictions .== labels)
        total += size(labels, 2)
    end
    
    return correct / total
end

"""
    evaluate_model(model, loss_fn, data_loader)

Evalúa el modelo en un conjunto de datos usando un DataLoader.
Devuelve la pérdida y exactitud.
"""
function evaluate_model(model, loss_fn, data_loader)
    total_loss = 0.0
    num_batches = 0
    correct = 0
    total = 0
    
    for (batch_x, batch_y) in data_loader
        # Forward pass
        output = NeuralNetwork.forward(model, batch_x)
        
        # Calcular loss
        loss = loss_fn(output, batch_y)
        total_loss += loss.data[1]
        num_batches += 1
        
        # Calcular exactitud
        predictions = argmax(output.data, dims=1)
        labels = argmax(batch_y.data, dims=1)
        correct += sum(predictions .== labels)
        total += size(labels, 2)
    end
    
    return total_loss / num_batches, correct / total
end

"""
Función mejorada de entrenamiento con augmentación y learning rate scheduler
"""
function train_improved!(
    model,
    optimizer,
    loss_fn,
    train_data,
    train_labels,
    epochs;
    batch_size=32,
    val_data=nothing,
    val_labels=nothing,
    lr_scheduler=nothing,
    use_augmentation=false,
    patience=10,
    verbose=true
)
    # Comprobar si podemos usar GPU
    use_gpu = CUDA.functional()
    
    if use_gpu && verbose
        println("Usando aceleración GPU")
        device_name = CUDA.name(CUDA.device())
        println("Dispositivo: $device_name")
    end
    
    # Inicializar resultados
    train_losses = Float64[]
    val_losses = Float64[]
    train_accs = Float64[]
    val_accs = Float64[]
    
    # Early stopping
    best_val_loss = Inf
    wait_epochs = 0
    best_weights = nothing
    
    # Parámetros del modelo
    params = NeuralNetwork.collect_parameters(model)
    
    for epoch in 1:epochs
        epoch_start = time()
        
        # Learning rate scheduling
        if lr_scheduler !== nothing
            if hasproperty(optimizer, :learning_rate)
                optimizer.learning_rate = lr_scheduler(optimizer.learning_rate, epoch)
                verbose && println("Epoch $epoch/$epochs (lr: $(optimizer.learning_rate)):")
            end
        else
            verbose && println("Epoch $epoch/$epochs:")
        end
        
        # ===== ENTRENAMIENTO =====
        # Barajar índices
        indices = shuffle(1:length(train_data))
        epoch_loss = 0.0
        num_batches = 0
        
        for i in 1:batch_size:length(indices)
            end_idx = min(i + batch_size - 1, length(indices))
            batch_indices = indices[i:end_idx]
            actual_batch_size = length(batch_indices)
            
            # Preparar lote
            try
                # Extraer imágenes y etiquetas
                batch_images = [train_data[j] for j in batch_indices]
                batch_labels = [train_labels[j] for j in batch_indices]
                
                # Convertir a tensores 4D si es necesario
                batch_tensors = Vector{Tensor}(undef, actual_batch_size)
                for j in 1:actual_batch_size
                    img = batch_images[j]
                    if ndims(img) == 3 || (isa(img, Tensor) && ndims(img.data) == 3)
                        # Convertir a formato NCHW (batch, channels, height, width)
                        if isa(img, Tensor)
                            img_data = img.data
                        else
                            img_data = img
                        end
                        c, h, w = size(img_data)
                        img_4d = reshape(img_data, (1, c, h, w))
                        batch_tensors[j] = Tensor(img_4d)
                    else
                        batch_tensors[j] = isa(img, Tensor) ? img : Tensor(img)
                    end
                end
                
                # Mover a GPU si es necesario
                if use_gpu
                    batch_tensors = [to_gpu(tensor) for tensor in batch_tensors]
                    batch_labels = [to_gpu(label) for label in batch_labels]
                end
                
                # Apilar tensores
                batch_x = stack_batch(batch_tensors)
                batch_y = stack_batch(batch_labels)
                
                # CAMBIO FASE 1: Usar zero_grad!
                for param in params
                    TensorEngine.zero_grad!(param)
                end
                
                # Forward pass
                output = NeuralNetwork.forward(model, batch_x)
                
                # Calcular loss
                loss = loss_fn(output, batch_y)
                
                if isnan(loss.data[1])
                    verbose && println("  ⚠️ Pérdida NaN detectada, saltando lote")
                    continue
                end
                
                # Backpropagation
                if loss.backward_fn !== nothing
                    grad = ones(size(loss.data))
                    if use_gpu
                        grad = CUDA.CuArray(grad)
                    end
                    loss.backward_fn(grad)
                end
                
                # Actualizar parámetros
                step!(optimizer, params)
                
                # Acumular pérdida
                epoch_loss += loss.data[1]
                num_batches += 1
                
                # Liberar memoria GPU
                if use_gpu
                    CUDA.unsafe_free!(batch_x.data)
                    CUDA.unsafe_free!(batch_y.data)
                    CUDA.unsafe_free!(output.data)
                    GC.gc()
                    CUDA.reclaim()
                end
            catch e
                verbose && println("  ⚠️ Error en lote: $e")
                continue
            end
        end
        
        # Calcular pérdida promedio
        epoch_loss = num_batches > 0 ? epoch_loss / num_batches : NaN
        push!(train_losses, epoch_loss)
        
        # ===== VALIDACIÓN =====
        val_loss = NaN
        val_acc = 0.0
        
        if val_data !== nothing && val_labels !== nothing
            try
                val_loss_sum = 0.0
                val_batches = 0
                
                # Validación por lotes
                for i in 1:batch_size:length(val_data)
                    end_idx = min(i + batch_size - 1, length(val_data))
                    val_indices = i:end_idx
                    
                    # Extraer imágenes y etiquetas
                    val_images = [val_data[j] for j in val_indices]
                    val_labels_batch = [val_labels[j] for j in val_indices]
                    
                    # Convertir a tensores 4D
                    val_tensors = Vector{Tensor}(undef, length(val_indices))
                    for j in 1:length(val_indices)
                        img = val_images[j]
                        if ndims(img) == 3 || (isa(img, Tensor) && ndims(img.data) == 3)
                            if isa(img, Tensor)
                                img_data = img.data
                            else
                                img_data = img
                            end
                            c, h, w = size(img_data)
                            img_4d = reshape(img_data, (1, c, h, w))
                            val_tensors[j] = Tensor(img_4d)
                        else
                            val_tensors[j] = isa(img, Tensor) ? img : Tensor(img)
                        end
                    end
                    
                    # Mover a GPU si es necesario
                    if use_gpu
                        val_tensors = [to_gpu(tensor) for tensor in val_tensors]
                        val_labels_batch = [to_gpu(label) for label in val_labels_batch]
                    end
                    
                    # Apilar tensores
                    val_x = stack_batch(val_tensors)
                    val_y = stack_batch(val_labels_batch)
                    
                    # Forward pass
                    val_output = NeuralNetwork.forward(model, val_x)
                    
                    # Calcular loss
                    val_batch_loss = loss_fn(val_output, val_y)
                    
                    if !isnan(val_batch_loss.data[1])
                        val_loss_sum += val_batch_loss.data[1]
                        val_batches += 1
                    end
                    
                    # Liberar memoria GPU
                    if use_gpu
                        CUDA.unsafe_free!(val_x.data)
                        CUDA.unsafe_free!(val_y.data)
                        CUDA.unsafe_free!(val_output.data)
                    end
                end
                
                val_loss = val_batches > 0 ? val_loss_sum / val_batches : NaN
                push!(val_losses, val_loss)
                
                # Calcular exactitud
                val_acc = compute_accuracy_batched(model, val_data, val_labels, batch_size=batch_size)
                push!(val_accs, val_acc)
                
                # Early stopping
                if !isnan(val_loss) && val_loss < best_val_loss
                    best_val_loss = val_loss
                    wait_epochs = 0
                    # Guardar mejores pesos
                    best_weights = deepcopy([param.data for param in params])
                    verbose && println("  ✓ Nuevo mejor modelo (val_loss: $(round(val_loss, digits=4)))")
                else
                    wait_epochs += 1
                    if wait_epochs >= patience
                        verbose && println("  Early stopping triggered at epoch $epoch")
                        # Restaurar mejores pesos
                        if best_weights !== nothing
                            for (param, weights) in zip(params, best_weights)
                                param.data = copy(weights)
                            end
                        end
                        break
                    end
                end
            catch e
                verbose && println("  ⚠️ Error en validación: $e")
            end
        end
        
        # Calcular exactitud en entrenamiento
        train_acc = compute_accuracy_batched(model, train_data, train_labels, batch_size=batch_size)
        push!(train_accs, train_acc)
        
        # Calcular tiempo de época
        epoch_time = time() - epoch_start
        
        # Reportar progreso
        if verbose
            println("  Pérdida: train=$(isnan(epoch_loss) ? "NaN" : round(epoch_loss, digits=4)), val=$(isnan(val_loss) ? "NaN" : round(val_loss, digits=4))")
            println("  Exactitud: train=$(isnan(train_acc) ? "NaN" : round(train_acc*100, digits=2))%, val=$(isnan(val_acc) ? "NaN" : round(val_acc*100, digits=2))%")
            println("  Tiempo: $(round(epoch_time, digits=2))s")
        end
        
        # Liberar memoria GPU
        if use_gpu
            GC.gc()
            CUDA.reclaim()
        end
    end
    
    return train_losses, val_losses, train_accs, val_accs
end

# Función auxiliar para cálculo eficiente de exactitud por lotes
function compute_accuracy_batched(model, inputs, targets; batch_size=32)
    use_gpu = CUDA.functional()
    
    correct = 0
    total = 0
    
    for i in 1:batch_size:length(inputs)
        end_idx = min(i + batch_size - 1, length(inputs))
        indices = i:end_idx
        
        # Preparar lote
        batch_images = [inputs[j] for j in indices]
        batch_labels = [targets[j] for j in indices]
        
        # Convertir a tensores 4D
        batch_tensors = Vector{Tensor}(undef, length(indices))
        for j in 1:length(indices)
            img = batch_images[j]
            if ndims(img) == 3 || (isa(img, Tensor) && ndims(img.data) == 3)
                if isa(img, Tensor)
                    img_data = img.data
                else
                    img_data = img
                end
                c, h, w = size(img_data)
                img_4d = reshape(img_data, (1, c, h, w))
                batch_tensors[j] = Tensor(img_4d)
            else
                batch_tensors[j] = isa(img, Tensor) ? img : Tensor(img)
            end
        end
        
        # Mover a GPU si es necesario
        if use_gpu
            batch_tensors = [to_gpu(tensor) for tensor in batch_tensors]
        end
        
        # Apilar tensores
        batch_x = stack_batch(batch_tensors)
        
        # Forward pass
        output = NeuralNetwork.forward(model, batch_x)
        
        # Calcular precisión
        # Obtener predicciones (índice del valor máximo)
        predictions = output.data
        
        # Si batch_size > 1, las predicciones están en la dimensión 2
        if size(predictions, 2) > 1
            for j in 1:length(indices)
                pred_class = argmax(predictions[:, j])[1]
                true_class = argmax(batch_labels[j].data)[1]
                correct += (pred_class == true_class)
            end
        else
            # Si batch_size = 1, las predicciones están en la dimensión 1
            pred_class = argmax(predictions)[1]
            true_class = argmax(batch_labels[1].data)[1]
            correct += (pred_class == true_class)
        end
        
        total += length(indices)
        
        # Liberar memoria GPU
        if use_gpu
            CUDA.unsafe_free!(batch_x.data)
            CUDA.unsafe_free!(output.data)
        end
    end
    
    return total > 0 ? correct / total : 0.0
end

"""
    train_improved_gpu!(model, optimizer, loss_fn, train_data, train_labels, epochs; kwargs...)

Versión de train_improved! específicamente optimizada para GPU usando DataLoaders.
"""
function train_improved_gpu!(
    model,
    optimizer,
    loss_fn,
    train_data,
    train_labels,
    epochs;
    batch_size=32,  # Batch size más grande para GPU
    val_data=nothing,
    val_labels=nothing,
    lr_scheduler=nothing,
    use_augmentation=true,
    patience=10,
    verbose=true
)
    verbose && println("Usando versión optimizada para GPU")
    
    # Crear DataLoaders optimizados
    train_loader = DataLoaders.optimized_data_loader(
        train_data,
        train_labels,
        batch_size,
        shuffle=true,
        to_gpu=true,
        prefetch=2
    )
    
    val_loader = nothing
    if val_data !== nothing && val_labels !== nothing
        val_loader = DataLoaders.optimized_data_loader(
            val_data,
            val_labels,
            batch_size,
            shuffle=false,
            to_gpu=true,
            prefetch=1
        )
    end
    
    # Inicializar historial
    train_losses = Float64[]
    val_losses = Float64[]
    train_accs = Float64[]
    val_accs = Float64[]
    
    # Inicializar early stopping
    best_val_loss = Inf
    wait_epochs = 0
    best_weights = nothing
    min_delta = 0.0001
    
    params = NeuralNetwork.collect_parameters(model)
    
    for epoch in 1:epochs
        epoch_start = time()
        
        # Actualizar learning rate si hay scheduler
        if lr_scheduler !== nothing
            current_lr = LRSchedulers.get_lr(lr_scheduler, epoch)
            optimizer.learning_rate = current_lr
            verbose && println("Epoch $epoch/$epochs (lr: $current_lr):")
        else
            verbose && println("Epoch $epoch/$epochs:")
        end
        
        # Training
        epoch_loss = 0.0
        num_batches = 0
        
        for (batch_x, batch_y) in train_loader
            # CAMBIO FASE 1: Usar zero_grad!
            for param in params
                TensorEngine.zero_grad!(param)
            end
            
            # Forward pass
            output = NeuralNetwork.forward(model, batch_x)
            
            # Calcular loss
            loss = loss_fn(output, batch_y)
            
            # Backpropagation
            if loss.backward_fn !== nothing
                loss.backward_fn(ones(size(loss.data)))
            end
            
            # Actualizar parámetros
            step!(optimizer, params)
            
            # Acumular pérdida
            epoch_loss += loss.data[1]
            num_batches += 1
            
            # Liberar tensores temporales
            if batch_x.data isa CUDA.CuArray
                GPUMemoryManager.release_tensor_buffer(batch_x.data)
            end
            if batch_y.data isa CUDA.CuArray
                GPUMemoryManager.release_tensor_buffer(batch_y.data)
            end
        end
        
        # Calcular pérdida promedio de la época
        epoch_loss /= num_batches
        push!(train_losses, epoch_loss)
        
        # Validación
        val_loss = NaN
        val_acc = 0.0
        
        if val_loader !== nothing
            # Evaluar en validación
            val_loss, val_acc = evaluate_model(model, loss_fn, val_loader)
            push!(val_losses, val_loss)
            push!(val_accs, val_acc)
            
            # Early stopping
            if val_loss < best_val_loss - min_delta
                best_val_loss = val_loss
                wait_epochs = 0
                # Guardar mejores pesos (deep copy)
                best_weights = deepcopy([param.data for param in params])
                verbose && println("  ✓ Nuevo mejor modelo (val_loss: $(round(val_loss, digits=4)))")
            else
                wait_epochs += 1
                if wait_epochs >= patience
                    verbose && println("Early stopping triggered at epoch $epoch")
                    # Restaurar mejores pesos
                    if best_weights !== nothing
                        for (param, weights) in zip(params, best_weights)
                            param.data = copy(weights)
                        end
                    end
                    break
                end
            end
        else
            # Si no hay datos de validación, usar la pérdida de entrenamiento
            push!(val_losses, epoch_loss)
        end
        
        # Calcular exactitud en entrenamiento
        train_acc = compute_accuracy_with_loader(model, train_loader)
        push!(train_accs, train_acc)
        
        # Calcular tiempo de época
        epoch_time = time() - epoch_start
        
        # Reportar progreso
        if verbose
            println("  Pérdida: train=$(round(epoch_loss, digits=4)), val=$(round(val_loss, digits=4))")
            println("  Exactitud: train=$(round(train_acc*100, digits=2))%, val=$(round(val_acc*100, digits=2))%")
            println("  Tiempo: $(round(epoch_time, digits=2))s")
        end
        
        # Liberar memoria GPU entre épocas
        if CUDA.functional()
            GPUMemoryManager.clear_cache()
            GC.gc()
            CUDA.reclaim()
        end
    end
    
    return train_losses, val_losses, train_accs, val_accs
end

"""
    adapt_image(img::TensorEngine.Tensor)

Adapta una imagen en formato (C, H, W) a formato (1, C, H, W) para procesamiento por redes.
"""
function adapt_image(img::TensorEngine.Tensor)
    data = img.data
    if ndims(data) == 3
        c, h, w = size(data)
        data_reshaped = reshape(data, (1, c, h, w))
    else
        data_reshaped = data
    end
    return TensorEngine.Tensor(data_reshaped)
end

end # module Training