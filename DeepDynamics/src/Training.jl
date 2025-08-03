module Training
using Printf
using ..TensorEngine
using ..NeuralNetwork
using ..Optimizers
using ..Visualizations
using ..GPUMemoryManager  # Nuevo import para gestión de memoria
using ..DataLoaders       # Nuevo import para DataLoaders optimizados
using ..Utils: set_training_mode!
using ..Callbacks: AbstractCallback, EarlyStopping, FinalReportCallback,
                   on_epoch_end, on_train_end
                   # Al inicio del módulo Training, después de using ..Callbacks
using ..Callbacks: on_epoch_begin, on_epoch_end, on_train_begin, on_train_end, 
                   on_batch_begin, on_batch_end, PrintCallback, ProgressCallback,
                   ReduceLROnPlateau
using ..Losses
using Random
using LinearAlgebra
using CUDA
using Statistics
using ..Utils
import ..Optimizers: step!
export train!, train_batch!, compute_accuracy_general, train_improved!,
       EarlyStopping, PrintCallback, FinalReportCallback, add_callback!,  # Estos ahora vienen de Callbacks
       run_epoch_callbacks, run_final_callbacks, 
       train_with_loaders, stack_batch, evaluate_model, 
       fit!, History
using ..Metrics: accuracy, mae, rmse, binary_accuracy
const optim_step! = Optimizers.step!
const AnyDataLoader = Union{DataLoader, DataLoaders.OptimizedDataLoader}   

using ..Logging: setup_logging
  

# -----------------------------------------------------------------------
# Estructuras de EarlyStopping, Callbacks, etc.
# -----------------------------------------------------------------------
const callbacks = AbstractCallback[]

# Función auxiliar para calcular métricas
function compute_metric(metric::Symbol, y_pred::Tensor, y_true::Tensor)
    y_pred_cpu = y_pred.data isa CUDA.CuArray ? Array(y_pred.data) : y_pred.data
    y_true_cpu = y_true.data isa CUDA.CuArray ? Array(y_true.data) : y_true.data
    
    if metric == :accuracy
        # Para clasificación multiclase
        if size(y_pred_cpu, 1) > 1
            pred_classes = vec(argmax(y_pred_cpu, dims=1))
            true_classes = vec(argmax(y_true_cpu, dims=1))
            return sum(pred_classes .== true_classes) / length(true_classes)
        else
            # Para clasificación binaria
            return binary_accuracy(vec(y_pred_cpu), vec(Int.(y_true_cpu .> 0.5)))
        end
    elseif metric == :mae
        return mae(vec(y_pred_cpu), vec(y_true_cpu))
    elseif metric == :rmse
        return rmse(vec(y_pred_cpu), vec(y_true_cpu))
    else
        # Si no se reconoce la métrica, devolver 0
        @warn "Métrica no reconocida: $metric"
        return 0.0f0
    end
end

function should_stop_early!(es::EarlyStopping, val_loss::Float64)
    # Crear logs falsos para usar la nueva interfaz
    logs = Dict(:val_loss => val_loss)
    on_epoch_end(es, 0, logs)  # epoch 0 es dummy
    return es.stopped
end



function add_callback!(cb)  # Sin especificar tipo
    push!(callbacks, cb)
end

# Función para limpiar callbacks si es necesario
function clear_callbacks!()
    empty!(callbacks)
end

# Funciones de compatibilidad actualizadas
function run_epoch_callbacks(epoch::Int, train_loss, val_loss, train_acc, val_acc)
    logs = Dict(
        :loss => train_loss,
        :val_loss => val_loss,
        :train_accuracy => train_acc,
        :val_accuracy => val_acc
    )
    
    for cb in callbacks
        try
            # Intentar la nueva interfaz
            on_epoch_end(cb, epoch, logs)
        catch
            # Si falla, intentar la interfaz antigua
            if hasproperty(cb, :on_epoch_end)
                on_epoch_end(cb, epoch, train_loss, val_loss, train_acc, val_acc)
            end
        end
    end
end

function run_final_callbacks(train_losses::Vector{Float64}, val_losses::Vector{Float64})
    logs = Dict(
        :train_losses => train_losses,
        :val_losses => val_losses
    )
    
    for cb in callbacks
        try
            on_train_end(cb, logs)
        catch
            # Compatibilidad con interfaz antigua si es necesario
            if cb isa FinalReportCallback
                on_training_end(cb, train_losses, val_losses)
            end
        end
    end
end



# Función auxiliar para asegurar que un tensor esté en el dispositivo correcto
function ensure_tensor_on_device(tensor::Tensor, device::Symbol)
    current_device = tensor.data isa CUDA.CuArray ? :gpu : :cpu
    if current_device != device
        if device == :gpu
            return TensorEngine.to_gpu(tensor)
        else
            return TensorEngine.to_cpu(tensor)
        end
    end
    return tensor
end


# Historia para almacenar métricas
mutable struct History
    train_loss::Vector{Float32}
    val_loss::Vector{Float32}
    train_metrics::Dict{String, Vector{Float32}}
    val_metrics::Dict{String, Vector{Float32}}
    epochs::Int
    
    History() = new(Float32[], Float32[], Dict{String, Vector{Float32}}(), 
                    Dict{String, Vector{Float32}}(), 0)
end

"""
    fit!(model, train_loader::DataLoader; kwargs...)

Entrena el modelo usando DataLoader con soporte completo para callbacks y métricas.
Soporta tanto DataLoader estándar como OptimizedDataLoader.
"""
function fit!(model::Sequential,
              train_loader::AnyDataLoader;
              val_loader::Union{AnyDataLoader, Nothing}=nothing,
              optimizer=Adam(0.001f0),
              loss_fn=mse_loss,
              epochs::Int=10,
              callbacks::Vector{<:AbstractCallback}=AbstractCallback[],
              verbose::Int=1,
              validation_data::Union{Nothing, Tuple{Any,Any}}=nothing,
              log_dir::Union{Nothing,String}=nothing,
              experiment_name::Union{Nothing,String}=nothing,
              log_config::Dict            = Dict(),   # ← sin parámetros de tipo
              use_tensorboard::Bool       = false,
              log_gradients::Bool         = false)


    batch_size = hasproperty(train_loader, :batch_size) ? 
                    train_loader.batch_size : 
                    error("No pude inferir batch_size desde $(typeof(train_loader))")

    # Si vienen datos de validación por este keyword, los usamos
    if validation_data !== nothing
        Xv, yv = validation_data
        val_loader = DataLoader(Xv, yv, batch_size; shuffle=false)
    end
        
    # Convertir callbacks a tipo correcto
    callbacks = Vector{AbstractCallback}(callbacks)
    
    # Agregar PrintCallback si verbose > 0
    if verbose > 0 && !any(cb isa PrintCallback for cb in callbacks)
        pushfirst!(callbacks, PrintCallback(1))
    end
    
    # Agregar ProgressCallback si verbose > 1
    if verbose > 1
        push!(callbacks, ProgressCallback(verbose))
    end
    if log_dir !== nothing && experiment_name !== nothing
        # Configurar experimento
        exp_config = merge(Dict{String, Any}(
            "model_type" => string(typeof(model)),
            "epochs" => epochs,
            "batch_size" => isdefined(train_loader, :batch_size) ? train_loader.batch_size : 32,
            "optimizer" => string(typeof(optimizer)),
            "learning_rate" => optimizer.learning_rate,
            "loss_fn" => string(loss_fn)
        ), Dict{String, Any}(log_config))  # ← Convertir log_config al tipo correcto
        
        # Crear tracker y callbacks de logging
        tracker, log_callbacks = setup_logging(
            experiment_name,
            exp_config,
            log_dir=log_dir,
            use_tensorboard=use_tensorboard,
            log_gradients=log_gradients
        )
        
        # Agregar callbacks de logging a la lista existente
        append!(callbacks, log_callbacks)
        
        if verbose > 0
            println("📊 Logging habilitado en: $(tracker.base_dir)")
        end
    end    
    # Inicializar historia
    history = History()
    params = collect_parameters(model)
    
    # Detectar dispositivo del modelo
    model_dev = model_device(model)
    
    # Logs globales para callbacks
    logs = Dict{Symbol, Any}(
        :model => model,
        :optimizer => optimizer,
        :params => params,
        :train_losses => Float64[],
        :val_losses => Float64[]
    )
    
    # on_train_begin
    for cb in callbacks
        on_train_begin(cb, logs)
    end
    
    # Configurar callbacks que necesitan referencias
    for cb in callbacks
        if cb isa ReduceLROnPlateau && hasproperty(cb, :optimizer_ref)
            cb.optimizer_ref = optimizer
        elseif cb isa ProgressCallback && hasproperty(cb, :total_batches)
            cb.total_batches = length(train_loader)
        end
    end
    
    # Loop principal de entrenamiento
    for epoch in 1:epochs
        epoch_start_time = time()
        
        # Logs de época
        epoch_logs = Dict{Symbol, Any}(:epoch => epoch)
        
        # on_epoch_begin
        for cb in callbacks
            on_epoch_begin(cb, epoch, epoch_logs)
        end
        
        # Establecer modo training
        set_training_mode!(model, true)
        
        # Métricas de época
        epoch_loss_sum = 0.0
        epoch_correct = 0
        epoch_total = 0
        batch_count = 0
        
        # Mostrar progreso si verbose > 0
        if verbose > 0
            println("\nÉpoca $epoch/$epochs")
            verbose > 1 && print("Entrenamiento: ")
        end
        
        # Iterar sobre batches del DataLoader
        for (batch_idx, (batch_x, batch_y)) in enumerate(train_loader)
            batch_logs = Dict{Symbol, Any}(:batch => batch_idx)
            
            # on_batch_begin
            for cb in callbacks
                on_batch_begin(cb, batch_idx, batch_logs)
            end
            
            # PROCESAMIENTO INTELIGENTE DE BATCHES
            # Necesitamos manejar 3 casos posibles:
            # 1. Vector de Tensores (DataLoader normal)
            # 2. Tensor 2D correcto (OptimizedDataLoader bien formateado)
            # 3. Tensor 1D aplanado (OptimizedDataLoader mal formateado)
            
            # Procesar input batch
            input_batch = process_batch_input(batch_x, model)
            
            # Procesar target batch - necesita saber las dimensiones esperadas
            target_batch = process_batch_target(batch_y, input_batch, loss_fn)
            
            # Actualizar tamaño del batch en logs
            batch_logs[:size] = size(input_batch.data, 2)
            
            # Sincronización GPU/CPU
            if model_dev == :gpu
                if !(input_batch.data isa CUDA.CuArray)
                    input_batch = to_gpu(input_batch)
                end
                if !(target_batch.data isa CUDA.CuArray)
                    target_batch = to_gpu(target_batch)
                end
            else
                if input_batch.data isa CUDA.CuArray
                    input_batch = to_cpu(input_batch)
                end
                if target_batch.data isa CUDA.CuArray
                    target_batch = to_cpu(target_batch)
                end
            end
            
            # Zero gradients
            for param in params
                zero_grad!(param)
            end
            
            # Forward pass
            output = forward(model, input_batch)
            
            # Calcular loss
            loss = loss_fn(output, target_batch)
            batch_loss = if loss.data isa CUDA.CuArray
                Array(loss.data)[1]
            else
                loss.data[1]
            end
            
            # Backward pass
            if loss.backward_fn !== nothing
                loss.backward_fn(ones(size(loss.data)))
            end
            
            # Actualizar parámetros
            optim_step!(optimizer, params)
            
            # Acumular métricas
            epoch_loss_sum += batch_loss
            batch_count += 1
            
            # Calcular accuracy si es clasificación
            if ndims(output.data) == 2 && size(output.data, 1) > 1
                # Multi-clase
                preds = argmax(output.data, dims=1)
                labels = argmax(target_batch.data, dims=1)
                epoch_correct += sum(preds .== labels)
                epoch_total += size(labels, 2)
            elseif ndims(output.data) == 2 && size(output.data, 1) == 1
                # Binaria
                preds = output.data .> 0.5
                labels = target_batch.data .> 0.5
                epoch_correct += sum(preds .== labels)
                epoch_total += length(labels)
            end
            
            batch_logs[:loss] = batch_loss
            
            # on_batch_end
            for cb in callbacks
                on_batch_end(cb, batch_idx, batch_logs)
            end
            
            # Limpieza periódica de memoria GPU
            if CUDA.functional() && batch_idx % 10 == 0
                CUDA.reclaim()
            end
        end
        
        # Calcular métricas finales de época
        train_loss = epoch_loss_sum / batch_count
        train_acc = epoch_total > 0 ? epoch_correct / epoch_total : 0.0
        
        # Guardar en historia
        push!(history.train_loss, Float32(train_loss))
        if !haskey(history.train_metrics, "accuracy")
            history.train_metrics["accuracy"] = Float32[]
        end
        push!(history.train_metrics["accuracy"], Float32(train_acc))
        
        # Evaluación en validación
        val_loss = NaN
        val_acc = 0.0
        if val_loader !== nothing
            set_training_mode!(model, false)
            val_loss, val_acc = evaluate_loader(model, loss_fn, val_loader, model_dev)
            push!(history.val_loss, Float32(val_loss))
            if !haskey(history.val_metrics, "accuracy")
                history.val_metrics["accuracy"] = Float32[]
            end
            push!(history.val_metrics["accuracy"], Float32(val_acc))
        else
            push!(history.val_loss, NaN32)
            if !haskey(history.val_metrics, "accuracy")
                history.val_metrics["accuracy"] = Float32[]
            end
            push!(history.val_metrics["accuracy"], NaN32)
        end
        
        # Imprimir resumen de época
        if verbose > 0
            if verbose > 1
                println()  # Nueva línea después del progress bar
            end
            elapsed = round(time() - epoch_start_time, digits=1)
            print("  $(elapsed)s - loss: $(round(train_loss, digits=4))")
            print(" - accuracy: $(round(train_acc*100, digits=2))%")
            if val_loader !== nothing && !isnan(val_loss)
                print(" - val_loss: $(round(val_loss, digits=4))")
                print(" - val_accuracy: $(round(val_acc*100, digits=2))%")
            end
            println()
        end
        
        # Preparar logs para callbacks de época
        epoch_logs[:loss] = train_loss
        epoch_logs[:train_accuracy] = train_acc
        epoch_logs[:val_loss] = val_loss
        epoch_logs[:val_accuracy] = val_acc
        epoch_logs[:time] = time() - epoch_start_time
        epoch_logs[:training_loss] = train_loss      # ← AGREGAR
        epoch_logs[:training_accuracy] = train_acc   # ← AGREGAR
        epoch_logs[:model] = model                   # ← AGREGAR (punto 3)
        # on_epoch_end para todos los callbacks
        should_stop = false
        for cb in callbacks
            on_epoch_end(cb, epoch, epoch_logs)
            
            # Verificar early stopping
            if cb isa EarlyStopping && hasproperty(cb, :stopped) && cb.stopped
                should_stop = true
            end
        end
        
        if should_stop
            verbose > 0 && println("\nEarly stopping en época $epoch")
            break
        end
        
        # Actualizar épocas en historia
        history.epochs = epoch
        
        # Memory cleanup al final de cada época
        if CUDA.functional()
            GPUMemoryManager.clear_cache()
            GC.gc()
            CUDA.reclaim()
        end
    end
    
    # on_train_end para todos los callbacks
    logs[:history] = history
    for cb in callbacks
        on_train_end(cb, logs)
    end
    
    # Establecer modo evaluación
    set_training_mode!(model, false)
    
    # Limpiar DataLoaders optimizados si aplica
    if isdefined(train_loader, :is_active) && train_loader.is_active
        cleanup_data_loader!(train_loader)
    end
    if val_loader !== nothing && isdefined(val_loader, :is_active) && val_loader.is_active
        cleanup_data_loader!(val_loader)
    end
    
    return history
end


function fit!(model::Sequential,
              X_train::Vector{<:Tensor}, y_train::Vector{<:Tensor};
              X_val=nothing,
              y_val=nothing,
              optimizer=Adam(0.001f0),
              loss_fn=mse_loss,
              epochs::Int=10,
              batch_size::Int=32,
              validation_split::Float32=0.2f0,
              shuffle::Bool=true,
              callbacks::AbstractVector{<:AbstractCallback}=AbstractCallback[],
              verbose::Bool=true,
              metrics::Vector{Symbol} = [:accuracy],
              validation_data::Union{Nothing, Tuple{Vector{<:Tensor},Vector{<:Tensor}}}=nothing,
              log_dir::Union{Nothing,String}=nothing,
              experiment_name::Union{Nothing,String}=nothing,
              log_config::Dict            = Dict(),   # ← sin parámetros de tipo
              use_tensorboard::Bool       = false,
              log_gradients::Bool         = false)

    # Desempaquetar validation_data si llega por este keyword
    if validation_data !== nothing
        X_val, y_val = validation_data
    end
    
    if log_dir !== nothing && experiment_name !== nothing
        # Configurar experimento
        exp_config = merge(Dict(
            "model_type" => string(typeof(model)),
            "epochs" => epochs,
            "batch_size" => batch_size,
            "optimizer" => string(typeof(optimizer)),
            "learning_rate" => optimizer.learning_rate,
            "loss_fn" => string(loss_fn)
        ), log_config)
        
        # Crear tracker y callbacks de logging
        tracker, log_callbacks = setup_logging(
            experiment_name,
            exp_config,
            log_dir=log_dir,
            use_tensorboard=use_tensorboard,
            log_gradients=log_gradients
        )
        
        # Agregar callbacks de logging a la lista existente
        append!(callbacks, log_callbacks)
        
        if verbose
            println("📊 Logging habilitado en: $(tracker.base_dir)")
        end
    end

    # Validación de inputs
    @assert length(X_train) == length(y_train) "X_train y y_train deben tener el mismo tamaño"
    @assert 0 <= validation_split < 1 "validation_split debe estar en [0, 1)"
    callbacks = Vector{AbstractCallback}(callbacks)
    # Si piden verbose, añadimos un PrintCallback al frente
        # Agregar PrintCallback si verbose es true y no hay ninguno
    if verbose && !any(cb isa PrintCallback for cb in callbacks)
        pushfirst!(callbacks, PrintCallback(1))
    end

    # Auto-split si no hay validación explícita
    if X_val === nothing && validation_split > 0
        n_samples = length(X_train)
        n_val = round(Int, n_samples * validation_split)
        n_train = n_samples - n_val
        
        indices = shuffle ? Random.shuffle(1:n_samples) : (1:n_samples)
        
        train_indices = indices[1:n_train]
        val_indices = indices[n_train+1:end]
        
        X_val = X_train[val_indices]
        y_val = y_train[val_indices]
        
        X_train_split = X_train[train_indices]
        y_train_split = y_train[train_indices]
    else
        X_train_split = X_train
        y_train_split = y_train
    end
    

    
    # Crear DataLoaders
    train_loader = DataLoader(X_train_split, y_train_split, batch_size; shuffle=shuffle)
    val_loader = if X_val !== nothing && y_val !== nothing
        DataLoader(X_val, y_val, batch_size; shuffle=false)
    else
        nothing
    end
    
    # Inicializar historia y parámetros
    history = History()
    params = collect_parameters(model)
    
    # Logs globales para callbacks
    logs = Dict{Symbol, Any}(
        :model => model,
        :optimizer => optimizer,
        :params => params,
        :train_losses => Float64[],
        :val_losses => Float64[]
    )
    logs[:history] = history
    # on_train_begin
    for cb in callbacks
        on_train_begin(cb, logs)
    end
    
    # Loop de entrenamiento
    for epoch in 1:epochs
        epoch_start_time = time()
        
        # Modo training
        set_training_mode!(model, true)
        
        # Logs de época - CORREGIDO
        epoch_logs = Dict{Symbol, Any}(
            :model => model,
            :optimizer => optimizer,
            :params => params,
            :epoch => epoch
        )
        
        # on_epoch_begin
        for cb in callbacks
            on_epoch_begin(cb, epoch, epoch_logs)
        end
        
        # === FASE DE ENTRENAMIENTO ===
        train_loss_sum = 0.0f0
        train_metric_sums = Dict{Symbol, Float32}(m => 0.0f0 for m in metrics)  # YA CORREGIDO
        train_batches = 0
        
        for (batch_idx, (batch_X, batch_y)) in enumerate(train_loader)
            # CORREGIDO: especificar tipo Any
            batch_logs = Dict{Symbol, Any}(:batch => batch_idx, :size => length(batch_X))
            
            # on_batch_begin
            for cb in callbacks
                on_batch_begin(cb, batch_idx, batch_logs)
            end
            
            # Zero gradients
            for p in params
                zero_grad!(p)
            end
            
            # Stack batch
            device = model_device(model)
            X_batch = stack_batch([ensure_tensor_on_device(x, device) for x in batch_X])
            y_batch = stack_batch([ensure_tensor_on_device(y, device) for y in batch_y])
            
            # Forward
            y_pred = forward(model, X_batch)
            
            # Loss
            loss = loss_fn(y_pred, y_batch)
            train_loss_sum += loss.data[1]
            
            # Backward
            backward(loss, [1.0f0])
            
            # Update
            optim_step!(optimizer, params)
            
            # Métricas
            for metric in metrics
                value = compute_metric(metric, y_pred, y_batch)
                train_metric_sums[metric] += value
            end
            
            train_batches += 1
            
            # on_batch_end - ESTA LÍNEA YA NO DARÁ ERROR
            batch_logs[:loss] = loss.data[1]
            for cb in callbacks
                on_batch_end(cb, batch_idx, batch_logs)
            end
        end
        
        # Promediar métricas de entrenamiento
        avg_train_loss = train_loss_sum / train_batches
        push!(history.train_loss, avg_train_loss)
        push!(logs[:train_losses], Float64(avg_train_loss))
        epoch_logs[:loss] = avg_train_loss
        epoch_logs[:training_loss] = avg_train_loss  # Para compatibilidad con TensorBoard


        for metric in metrics
            avg_value = train_metric_sums[metric] / train_batches
            if !haskey(history.train_metrics, string(metric))
                history.train_metrics[string(metric)] = Float32[]
            end
            push!(history.train_metrics[string(metric)], avg_value)
            
            # Usar nombres estándar para métricas comunes
            if metric == :accuracy
                epoch_logs[:training_accuracy] = avg_value  # ← AQUÍ, DENTRO del bucle
            else
                epoch_logs[Symbol("train_$metric")] = avg_value
            end
        end
        
        # === FASE DE VALIDACIÓN ===
        if val_loader !== nothing
            set_training_mode!(model, false)
            
            val_loss_sum = 0.0f0
            val_metric_sums = Dict{Symbol, Float32}(m => 0.0f0 for m in metrics)  # YA CORREGIDO
            val_batches = 0
            
            for (batch_X, batch_y) in val_loader
                # Stack batch - CORREGIDO
                device = model_device(model)
                X_batch = stack_batch([ensure_tensor_on_device(x, device) for x in batch_X])
                y_batch = stack_batch([ensure_tensor_on_device(y, device) for y in batch_y])
                
                # Forward sin gradientes
                y_pred = forward(model, X_batch)
                
                # Loss
                loss = loss_fn(y_pred, y_batch)
                val_loss_sum += loss.data[1]
                
                # Métricas
                for metric in metrics
                    value = compute_metric(metric, y_pred, y_batch)
                    val_metric_sums[metric] += value
                end
                
                val_batches += 1
            end
            
            # Promediar métricas de validación
            avg_val_loss = val_loss_sum / val_batches
            push!(history.val_loss, avg_val_loss)
            push!(logs[:val_losses], Float64(avg_val_loss))
            epoch_logs[:val_loss] = avg_val_loss
            
            for metric in metrics
                avg_value = val_metric_sums[metric] / val_batches  # ← val_metric_sums, NO train_metric_sums
                if !haskey(history.val_metrics, string(metric))
                    history.val_metrics[string(metric)] = Float32[]
                end
                push!(history.val_metrics[string(metric)], avg_value)
                
                if metric == :accuracy
                    epoch_logs[:val_accuracy] = avg_value
                else
                    epoch_logs[Symbol("val_$metric")] = avg_value
                end
            end
        end
        
        # Tiempo de época
        epoch_time = time() - epoch_start_time
        epoch_logs[:time] = epoch_time
        
        # on_epoch_end
        for cb in callbacks
            on_epoch_end(cb, epoch, epoch_logs)
        end
        
        # Verificar early stopping
        for cb in callbacks
            if cb isa EarlyStopping && cb.stopped
                history.epochs = epoch
                # on_train_end para todos los callbacks
                for cb2 in callbacks
                    on_train_end(cb2, logs)
                end
                return history
            end
        end
        
        history.epochs = epoch
        
        # Limpiar memoria GPU
        if CUDA.functional()
            GPUMemoryManager.clear_cache()
            GC.gc()
            CUDA.reclaim()
        end
    end
    
    # on_train_end
    for cb in callbacks
        on_train_end(cb, logs)
    end
    
    return history
end





# Función auxiliar para procesar input batches
function process_batch_input(batch_x, model)
    if isa(batch_x, Vector{<:TensorEngine.Tensor})
        # DataLoader normal: stack de vectores
        return stack_batch(batch_x)
    elseif isa(batch_x, TensorEngine.Tensor)
        # OptimizedDataLoader: ya es un tensor
        data = batch_x.data
        
        # Verificar si las dimensiones son correctas
        if ndims(data) == 1
            # Está aplanado, necesitamos reconstruir
            # Obtener el tamaño de entrada esperado de la primera capa
            first_layer = model.layers[1]
            if isa(first_layer, Dense)
                input_dim = size(first_layer.weights.data, 2)
                batch_size = div(length(data), input_dim)
                
                if length(data) != input_dim * batch_size
                    error("Datos mal formateados: $(length(data)) elementos no divisibles por input_dim=$input_dim")
                end
                
                # Reshape a (input_dim, batch_size)
                reshaped = reshape(data, input_dim, batch_size)
                return TensorEngine.Tensor(reshaped)
            else
                error("Primera capa no es Dense, no se puede inferir dimensiones")
            end
        elseif ndims(data) == 2
            # Ya tiene forma correcta
            return batch_x
        else
            error("Dimensiones inesperadas en batch_x: $(ndims(data))D")
        end
    else
        error("Tipo de batch no soportado: $(typeof(batch_x))")
    end
end

# Función auxiliar para procesar target batches
function process_batch_target(batch_y, input_batch, loss_fn)
    batch_size = size(input_batch.data, 2)
    
    if isa(batch_y, Vector{<:TensorEngine.Tensor})
        # DataLoader normal: stack de vectores
        return stack_batch(batch_y)
    elseif isa(batch_y, TensorEngine.Tensor)
        # OptimizedDataLoader: ya es un tensor
        data = batch_y.data
        
        if ndims(data) == 1
            # Está aplanado, necesitamos reconstruir
            # Para clasificación, asumimos one-hot encoding
            total_elements = length(data)
            
            if total_elements % batch_size != 0
                error("Target mal formateado: $total_elements elementos no divisibles por batch_size=$batch_size")
            end
            
            n_classes = div(total_elements, batch_size)
            
            # Reshape a (n_classes, batch_size)
            reshaped = reshape(data, n_classes, batch_size)
            return TensorEngine.Tensor(reshaped)
        elseif ndims(data) == 2
            # Ya tiene forma correcta
            return batch_y
        else
            error("Dimensiones inesperadas en batch_y: $(ndims(data))D")
        end
    else
        error("Tipo de batch no soportado: $(typeof(batch_y))")
    end
end

# evaluate_loader completo con la misma lógica
function evaluate_loader(model, loss_fn, data_loader::AnyDataLoader, device::Symbol)
    total_loss = 0.0
    correct = 0
    total = 0
    num_batches = 0
    
    for (batch_x, batch_y) in data_loader
        # Usar las mismas funciones de procesamiento
        input_batch = process_batch_input(batch_x, model)
        target_batch = process_batch_target(batch_y, input_batch, loss_fn)
        
        # Mover al dispositivo correcto
        if device == :gpu
            if !(input_batch.data isa CUDA.CuArray)
                input_batch = to_gpu(input_batch)
            end
            if !(target_batch.data isa CUDA.CuArray)
                target_batch = to_gpu(target_batch)
            end
        else
            if input_batch.data isa CUDA.CuArray
                input_batch = to_cpu(input_batch)
            end
            if target_batch.data isa CUDA.CuArray
                target_batch = to_cpu(target_batch)
            end
        end
        
        # Forward pass sin gradientes
        output = forward(model, input_batch)
        loss = loss_fn(output, target_batch)
        
        # Acumular loss
        loss_val = if loss.data isa CUDA.CuArray
            Array(loss.data)[1]
        else
            loss.data[1]
        end
        total_loss += loss_val
        num_batches += 1
        
        # Calcular accuracy basado en las dimensiones de salida
        if ndims(output.data) == 2
            if size(output.data, 1) > 1
                # Multi-clase
                preds = argmax(output.data, dims=1)
                labels = argmax(target_batch.data, dims=1)
                correct += sum(preds .== labels)
                total += size(labels, 2)
            else
                # Binaria (1 neurona de salida)
                preds = output.data .> 0.5
                labels = target_batch.data .> 0.5
                correct += sum(preds .== labels)
                total += length(labels)
            end
        end
    end
    
    avg_loss = num_batches > 0 ? total_loss / num_batches : NaN
    accuracy = total > 0 ? correct / total : 0.0
    
    return avg_loss, accuracy
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

"""
    safe_model_eval!(model)

Pone el modelo en modo evaluación de forma segura.
"""
function safe_model_eval!(model)
    try
        set_training_mode!(model, false)
    catch e
        @warn "No se pudo cambiar a modo eval: $e"
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
    TensorEngine.backward(loss, ones(size(loss.data)))
    
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
                
                val_loss_sum += Array(val_batch_loss.data)[1]

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

# Agregar al módulo Training en Training.jl

"""
    train!(model, X, y; kwargs...) -> Dict

Entrena un modelo con datos X,y usando configuración especificada.

# Argumentos
- `model`: Modelo Sequential a entrenar
- `X`: Datos de entrada (Array o Vector de Tensors)
- `y`: Etiquetas objetivo (Array o Vector de Tensors)

# Kwargs
- `optimizer`: Optimizador a usar (default: Adam(learning_rate=0.001))
- `loss_fn`: Función de pérdida (default: mse_loss)
- `epochs`: Número de épocas (default: 10)
- `batch_size`: Tamaño del batch (default: 32)
- `verbose`: Mostrar progreso (default: true)
- `metrics`: Vector de métricas a calcular (default: [])

# Retorna
Dict con historial: {loss: Vector{Float64}, metrics: Dict{Symbol, Vector{Float64}}}
"""
function train!(model, X, y; 
    optimizer=nothing,
    loss_fn=TensorEngine.mse_loss,
    epochs::Int=10,
    batch_size::Int=32,
    verbose::Bool=true,
    metrics::Vector=[])
    
    # Si no se proporciona optimizador, usar Adam por defecto
    if optimizer === nothing
        optimizer = Optimizers.Adam(learning_rate=0.001)
    end
    
    # Validación de entrada
    _validate_training_inputs(X, y)
    
    # Preparar datos como tensores
    X_tensors = _prepare_tensors(X)
    y_tensors = _prepare_tensors(y)
    
    # Detectar dispositivo del modelo
    model_dev = NeuralNetwork.model_device(model)
    use_gpu = (model_dev == :gpu)
    
    # Mover datos al dispositivo correcto
    if use_gpu
        X_tensors = [TensorEngine.to_gpu(t) for t in X_tensors]
        y_tensors = [TensorEngine.to_gpu(t) for t in y_tensors]
    end
    
    # ------------------------------------------------------
    # Para clasificación binaria con logits, usar full-batch
    # ------------------------------------------------------
    n_samples = length(X_tensors)
    if loss_fn === binary_crossentropy_with_logits
        batch_size = n_samples
    end
    # Recolectar parámetros
    params = NeuralNetwork.collect_parameters(model)
    
    # Inicializar historial
    history = Dict(
        :loss => Float64[],
        :metrics => Dict{Symbol, Vector{Float64}}()
    )
    
    # Inicializar métricas en historial
    for metric in metrics
        metric_name = Symbol(nameof(metric))
        history[:metrics][metric_name] = Float64[]
    end
    
    # Calcular número de batches
     n_batches = cld(n_samples, batch_size)

    
    # Training loop
    for epoch in 1:epochs
        epoch_start = time()
        epoch_loss = 0.0
        batch_count = 0
        
        # Shuffle de índices
        indices = Random.shuffle(1:n_samples)
        
        # Progress tracking
        verbose && println("\nEpoch $epoch/$epochs")
        
        # Procesar batches
        for batch_idx in 1:n_batches
            # Obtener índices del batch
            start_idx = (batch_idx - 1) * batch_size + 1
            end_idx = min(batch_idx * batch_size, n_samples)
            batch_indices = indices[start_idx:end_idx]
            
            # Preparar batch - usar Vector{Tensor} como espera train_batch!
            batch_X_tensors = X_tensors[batch_indices]
            batch_y_tensors = y_tensors[batch_indices]
            
            batch_X = stack_batch(batch_X_tensors)
            batch_y = stack_batch(batch_y_tensors)

            # Limpiar gradientes
            for param in params
                TensorEngine.zero_grad!(param)
            end

            # Forward pass
            output = NeuralNetwork.forward(model, batch_X)

            # Compute loss: si es BCE tras Activation, usar logits‐BCE
            if loss_fn === binary_crossentropy &&
               isa(model, Sequential) &&
               isa(model.layers[end], Activation)

                # submodelo sin la última sigmoid
                submodel = Sequential(model.layers[1:end-1])
                logits   = NeuralNetwork.forward(submodel, batch_X)
                loss     = binary_crossentropy_with_logits(logits, batch_y)
            else
                loss     = loss_fn(output, batch_y)
            end
            batch_loss = loss.data[1]

            # Backward + update
            TensorEngine.backward(loss, ones(size(loss.data)))
            step!(optimizer, params)
            
            epoch_loss += batch_loss
            batch_count += 1
            
            progress_interval = max(1, n_batches ÷ 20)

            if verbose && batch_idx % progress_interval == 0
                progress = batch_idx / n_batches * 100
                progress_blocks = round(Int, progress / 5)           # ✅ seguro
                remaining_blocks = 20 - progress_blocks              # ✅ seguro

                print("\rProgress: [" * "="^progress_blocks * " "^remaining_blocks * "] " *
                    Printf.@sprintf("%.1f%% | Loss: %.4f", progress, batch_loss))
            end

            
            # Limpiar memoria GPU si es necesario
            if use_gpu && batch_idx % 10 == 0
                GPUMemoryManager.check_and_clear_gpu_memory()
            end
        end
        
        # Calcular loss promedio de la época
        avg_epoch_loss = epoch_loss / batch_count
        push!(history[:loss], avg_epoch_loss)
        
        # Calcular métricas
        if !isempty(metrics)
            metric_values = _compute_epoch_metrics(
                model, X_tensors, y_tensors, metrics, batch_size, use_gpu
            )
            
            # Guardar métricas
            for (metric_name, value) in metric_values
                push!(history[:metrics][metric_name], value)
            end
            
            # Mostrar resumen de época
            if verbose
                elapsed = time() - epoch_start
                metric_str = join(["$k: $(round(v, digits=4))" for (k,v) in metric_values], ", ")
                println(Printf.@sprintf("\nEpoch %d/%d - %.2fs - loss: %.4f - %s", 
                        epoch, epochs, elapsed, avg_epoch_loss, metric_str))
            end
        else
            if verbose
                elapsed = time() - epoch_start
                println(Printf.@sprintf("\nEpoch %d/%d - %.2fs - loss: %.4f", 
                        epoch, epochs, elapsed, avg_epoch_loss))
            end
        end
    end
    
    return history
end

# Funciones auxiliares

function _validate_training_inputs(X, y)
    if isempty(X) || isempty(y)
        throw(ArgumentError("Los datos de entrada X e y no pueden estar vacíos"))
    end
    
    n_x = isa(X, AbstractArray) ? (ndims(X) > 1 ? size(X)[end] : length(X)) : length(X)
    n_y = isa(y, AbstractArray) ? (ndims(y) > 1 ? size(y)[end] : length(y)) : length(y)
    
    if n_x != n_y
        throw(DimensionMismatch("X tiene $n_x muestras pero y tiene $n_y muestras"))
    end
end

function _prepare_tensors(data)
    if isa(data, Vector{<:Tensor})
        return data
    elseif isa(data, AbstractArray)
        # Si es un array multidimensional, separar por última dimensión
        if ndims(data) > 1
            n_samples = size(data)[end]
            slices = [selectdim(data, ndims(data), i) for i in 1:n_samples]
            return [Tensor(Float32.(slice)) for slice in slices]
        else
            # Vector simple - IMPORTANTE: mantener como matriz columna
            return [Tensor(reshape(Float32.([x]), :, 1)) for x in data]
        end
    else
        throw(ArgumentError("Tipo de datos no soportado: $(typeof(data))"))
    end
end

function _compute_epoch_metrics(model, X_tensors, y_tensors, metrics, batch_size, use_gpu)
    metric_values = Pair{Symbol, Float64}[]
    
    n_samples = length(X_tensors)
    
    for metric in metrics
        metric_name = Symbol(nameof(metric))
        metric_sum = 0.0
        
        # Modo evaluación
        Utils.set_training_mode!(model, false)
        
        # Calcular métrica en todos los datos
        for i in 1:batch_size:n_samples
            end_idx = min(i + batch_size - 1, n_samples)
            batch_X = X_tensors[i:end_idx]
            batch_y = y_tensors[i:end_idx]
            
            # Stack batch
            batch_X_stacked = stack_batch(batch_X)
            batch_y_stacked = stack_batch(batch_y)
            
            # Forward pass
            predictions = NeuralNetwork.forward(model, batch_X_stacked)
            
            # Calcular métrica
            pred_data = predictions.data isa CUDA.CuArray ? Array(predictions.data) : predictions.data
            true_data = batch_y_stacked.data isa CUDA.CuArray ? Array(batch_y_stacked.data) : batch_y_stacked.data
            batch_metric = metric(pred_data, true_data)
            metric_sum += batch_metric * length(i:end_idx)
        end
        
        # Volver a modo entrenamiento
        Utils.set_training_mode!(model, true)
        
        avg_metric = metric_sum / n_samples
        push!(metric_values, metric_name => avg_metric)
    end
    
    return metric_values
end
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
            TensorEngine.backward(loss, ones(size(loss.data)))
            
            # Actualizar parámetros
            step!(optimizer, params)
            
            # Acumular pérdida
            epoch_loss += Array(loss.data)[1]
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
    # Establecer modo training
    set_training_mode!(model, true)
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
                
                if isnan(Array(loss.data)[1])
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
                epoch_loss += Array(loss.data)[1]
                num_batches += 1
                
                # Liberar memoria GPU
                if use_gpu
                    # Liberar buffers específicos del batch
                    try
                        # Intentar liberar con GPUMemoryManager
                        GPUMemoryManager.release_tensor_buffer(batch_x.data)
                        GPUMemoryManager.release_tensor_buffer(output.data)
                    catch
                        # Fallback a liberación directa si falla
                        CUDA.unsafe_free!(batch_x.data)
                        CUDA.unsafe_free!(output.data)
                    end
                    
                    # Auto-limpieza si es necesario
                    GPUMemoryManager.check_and_clear_gpu_memory(verbose=false)
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
                    
                    val_loss_val = Array(val_batch_loss.data)[1]
                    if !isnan(val_loss_val)
                        val_loss_sum += val_loss_val
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
        set_training_mode!(model, false)
    
    # Limpieza final de GPU
    if use_gpu
        GPUMemoryManager.clear_cache()
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
            TensorEngine.backward(loss, ones(size(loss.data)))
            
            # Actualizar parámetros
            step!(optimizer, params)
            
            # Acumular pérdida
            epoch_loss += Array(loss.data)[1]
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