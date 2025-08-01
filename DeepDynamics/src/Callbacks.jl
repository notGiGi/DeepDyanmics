module Callbacks

using ..TensorEngine
using ..NeuralNetwork
using ..Optimizers
#using ..Reports  # Para FinalReportCallback
using JLD2
using Printf

export AbstractCallback, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint,
       PrintCallback, FinalReportCallback,
       on_epoch_begin, on_epoch_end, on_train_begin, on_train_end,
       on_batch_begin, on_batch_end, ProgressCallback

# ==================================================================
# Interfaz base para callbacks
# ==================================================================
abstract type AbstractCallback end

# Métodos por defecto (no hacen nada)
on_epoch_begin(cb::AbstractCallback, epoch::Int, logs::Dict) = nothing
on_epoch_end(cb::AbstractCallback, epoch::Int, logs::Dict) = nothing
on_train_begin(cb::AbstractCallback, logs::Dict) = nothing
on_train_end(cb::AbstractCallback, logs::Dict) = nothing
on_batch_begin(cb::AbstractCallback, batch::Int, logs::Dict) = nothing
on_batch_end(cb::AbstractCallback, batch::Int, logs::Dict) = nothing

# ==================================================================
# PrintCallback (actualizado para nueva interfaz)
# ==================================================================
struct PrintCallback <: AbstractCallback
    freq::Int
    PrintCallback(freq::Int=1) = new(freq)
end

function on_epoch_end(cb::PrintCallback, epoch::Int, logs::Dict)
    if epoch % cb.freq == 0
        # Extraer métricas del diccionario logs
        train_loss = get(logs, :loss, NaN)
        val_loss = get(logs, :val_loss, NaN)
        train_acc = get(logs, :train_accuracy, get(logs, Symbol("train_accuracy"), NaN))
        val_acc = get(logs, :val_accuracy, get(logs, Symbol("val_accuracy"), NaN))
        
        print("Epoch $epoch: ")
        print("Training Loss = $(round(train_loss, digits=4))")
        
        if !isnan(val_loss)
            print(", Validation Loss = $(round(val_loss, digits=4))")
        end
        
        if !isnan(train_acc)
            print(", Training Accuracy = $(round(train_acc*100, digits=2))%")
        end
        
        if !isnan(val_acc)
            print(", Validation Accuracy = $(round(val_acc*100, digits=2))%")
        end
        
        println()
    end
end

# ==================================================================
# FinalReportCallback (actualizado para nueva interfaz)
# ==================================================================
struct FinalReportCallback <: AbstractCallback
    output_format::Symbol
    filename::String
    
    FinalReportCallback(; output_format::Symbol=:html, filename::String="report.html") = 
        new(output_format, filename)
end

function on_train_end(cb::FinalReportCallback, logs::Dict)
    # Obtener historias de pérdidas
    train_losses = get(logs, :train_losses, Float64[])
    val_losses = get(logs, :val_losses, Float64[])
    
    if !isempty(train_losses)
        Reports.generate_report(train_losses, val_losses; 
                              output_format=cb.output_format, 
                              filename=cb.filename)
    end
end

# ==================================================================
# EarlyStopping
# ==================================================================
mutable struct EarlyStopping <: AbstractCallback
    monitor::String
    patience::Int
    min_delta::Float32
    mode::Symbol
    restore_best_weights::Bool
    verbose::Bool
    
    # Estado interno
    wait::Int
    stopped_epoch::Int
    best::Float32
    best_weights::Union{Nothing, Vector}
    stopped::Bool
    
    function EarlyStopping(;
        monitor::String="val_loss",
        patience::Int=5,
        min_delta::Real=0.001,
        mode::Symbol=:min,
        restore_best_weights::Bool=true,
        verbose::Bool=true
    )
        @assert mode in [:min, :max] "mode debe ser :min o :max"
        best = mode == :min ? Inf32 : -Inf32
        new(monitor, patience, min_delta, mode, restore_best_weights, 
            verbose, 0, 0, best, nothing, false)
    end
end

function on_train_begin(cb::EarlyStopping, logs::Dict)
    # Reiniciar estado
    cb.wait = 0
    cb.stopped_epoch = 0
    cb.best = cb.mode == :min ? Inf32 : -Inf32
    cb.best_weights = nothing
    cb.stopped = false
end

function on_epoch_end(cb::EarlyStopping, epoch::Int, logs::Dict)
    current = get(logs, Symbol(cb.monitor), nothing)
    if current === nothing
        @warn "Early stopping: métrica '$(cb.monitor)' no encontrada"
        return
    end
    
    # Verificar si mejoró
    improved = if cb.mode == :min
        current < cb.best - cb.min_delta
    else
        current > cb.best + cb.min_delta
    end
    
    if improved
        cb.best = current
        cb.wait = 0
        
        # Guardar mejores pesos
        if cb.restore_best_weights && haskey(logs, :model)
            model = logs[:model]
            params = NeuralNetwork.collect_parameters(model)
            cb.best_weights = deepcopy([p.data for p in params])
        end
    else
        cb.wait += 1
        if cb.wait >= cb.patience
            cb.stopped_epoch = epoch
            cb.stopped = true
            
            if cb.verbose
                println("\nEpoch $epoch: early stopping")
            end
            
            # Restaurar mejores pesos
            if cb.restore_best_weights && cb.best_weights !== nothing && haskey(logs, :model)
                model = logs[:model]
                params = NeuralNetwork.collect_parameters(model)
                for (param, best_weight) in zip(params, cb.best_weights)
                    param.data .= best_weight
                end
                
                if cb.verbose
                    println("Restaurando pesos del mejor modelo")
                end
            end
        end
    end
end

# ==================================================================
# ReduceLROnPlateau
# ==================================================================
mutable struct ReduceLROnPlateau <: AbstractCallback
    monitor::String
    patience::Int
    factor::Float32
    min_lr::Float32
    mode::Symbol
    cooldown::Int
    verbose::Bool

    # Estado interno
    wait::Int
    best::Float32
    cooldown_counter::Int

    function ReduceLROnPlateau(;
        monitor::String="val_loss",
        patience::Int=10,
        factor::Real=0.1f0,      # aceptar Float32, Float64, Int…
        min_lr::Real=1e-7,
        mode::Symbol=:min,
        cooldown::Int=0,
        verbose::Bool=true
    )
        @assert mode in [:min, :max] "mode debe ser :min o :max"
        @assert 0 < factor < 1 "factor debe estar entre 0 y 1"
        # convierte a Float32 internamente
        f32  = Float32(factor)
        ml32 = Float32(min_lr)
        best = mode == :min ? Inf32 : -Inf32
        # aquí usamos f32 y ml32
        new(monitor, patience, f32, ml32, mode, cooldown,
            verbose, 0, best, 0)
    end
end

function on_train_begin(cb::ReduceLROnPlateau, logs::Dict)
    cb.wait = 0
    cb.best = cb.mode == :min ? Inf32 : -Inf32
    cb.cooldown_counter = 0
end

function on_epoch_end(cb::ReduceLROnPlateau, epoch::Int, logs::Dict)
    current = get(logs, Symbol(cb.monitor), nothing)
    if current === nothing
        @warn "ReduceLROnPlateau: métrica '$(cb.monitor)' no encontrada"
        return
    end
    
    # Si está en cooldown, decrementar contador
    if cb.cooldown_counter > 0
        cb.cooldown_counter -= 1
        cb.wait = 0
        return
    end
    
    # Verificar si mejoró
    improved = if cb.mode == :min
        current < cb.best
    else
        current > cb.best
    end
    
    if improved
        cb.best = current
        cb.wait = 0
    else
        cb.wait += 1
        if cb.wait >= cb.patience
            # Reducir learning rate
            optimizer = get(logs, :optimizer, nothing)
            if optimizer !== nothing && hasproperty(optimizer, :learning_rate)
                old_lr = optimizer.learning_rate
                new_lr = max(old_lr * cb.factor, cb.min_lr)
                
                if new_lr < old_lr
                    optimizer.learning_rate = new_lr
                    cb.wait = 0
                    cb.cooldown_counter = cb.cooldown
                    
                    if cb.verbose
                        println("\nEpoch $epoch: reduciendo learning rate de $old_lr a $new_lr")
                    end
                else
                    if cb.verbose
                        println("\nEpoch $epoch: learning rate ya en mínimo ($cb.min_lr)")
                    end
                end
            end
        end
    end
end

# ==================================================================
# ModelCheckpoint
# ==================================================================
mutable struct ModelCheckpoint <: AbstractCallback
    filepath::String
    monitor::String
    mode::Symbol
    save_best_only::Bool
    save_weights_only::Bool
    verbose::Bool
     
    # Estado interno
    best::Float32
    
    function ModelCheckpoint(
        filepath::String;
        monitor::String="val_loss",
        mode::Symbol=:min,
        save_best_only::Bool=true,
        save_weights_only::Bool=false,
        verbose::Bool=true
    )
        @assert mode in [:min, :max] "mode debe ser :min o :max"
        best = mode == :min ? Inf32 : -Inf32
        new(filepath, monitor, mode, save_best_only, save_weights_only, 
            verbose, best)
    end
end

function on_train_begin(cb::ModelCheckpoint, logs::Dict)
    cb.best = cb.mode == :min ? Inf32 : -Inf32
end

function on_epoch_end(cb::ModelCheckpoint, epoch::Int, logs::Dict)
    current = get(logs, Symbol(cb.monitor), nothing)
    if current === nothing
        @warn "ModelCheckpoint: métrica '$(cb.monitor)' no encontrada"
        return
    end
    
    # Verificar si debe guardar
    should_save = if cb.save_best_only
        improved = cb.mode == :min ? current < cb.best : current > cb.best
        if improved
            cb.best = current
            true
        else
            false
        end
    else
        true  # Guardar siempre si save_best_only es false
    end
    
    if should_save && haskey(logs, :model)
        model = logs[:model]
        
        # Formatear nombre del archivo
        filepath = cb.filepath
        filepath = replace(filepath, "{epoch}" => @sprintf("%03d", epoch))
        filepath = replace(filepath, "{$(cb.monitor)}" => @sprintf("%.4f", current))
        
        # Por compatibilidad con métricas adicionales
        for (key, value) in logs
            if isa(value, Number)
                placeholder = "{$key}"
                if occursin(placeholder, filepath)
                    filepath = replace(filepath, placeholder => @sprintf("%.4f", value))
                end
            end
        end
        
        # Guardar modelo o solo pesos
        try
            if cb.save_weights_only
                params = NeuralNetwork.collect_parameters(model)
                weights_dict = Dict{String, Array}()
                for (i, param) in enumerate(params)
                    weights_dict["param_$i"] = Array(param.data)
                end
                JLD2.save(filepath, "weights", weights_dict)
            else
                # Guardar modelo completo (simplificado por ahora)
                save_model_state(model, filepath)
            end
            
            if cb.verbose
                println("\nEpoch $epoch: guardando modelo en '$filepath'")
            end
        catch e
            @warn "Error guardando modelo: $e"
        end
    end
end

# Función auxiliar para guardar estado del modelo
function save_model_state(model::NeuralNetwork.Sequential, filepath::String)
    # Crear diccionario con la estructura del modelo
    model_dict = Dict{String, Any}()
    
    # Guardar arquitectura
    model_dict["num_layers"] = length(model.layers)
    
    # Guardar parámetros
    params = NeuralNetwork.collect_parameters(model)
    weights = Dict{String, Array}()
    for (i, param) in enumerate(params)
        # Convertir a CPU si está en GPU
        param_data = param.data isa CUDA.CuArray ? Array(param.data) : param.data
        weights["param_$i"] = param_data
    end
    model_dict["weights"] = weights
    
    # Información adicional de las capas
    layer_info = []
    for (i, layer) in enumerate(model.layers)
        info = Dict{String, Any}("type" => string(typeof(layer)))
        
        # Agregar información específica según el tipo de capa
        if layer isa NeuralNetwork.Dense
            info["input_size"] = size(layer.weights.data, 2)
            info["output_size"] = size(layer.weights.data, 1)
        elseif layer isa Layers.BatchNorm
            info["num_features"] = length(layer.gamma.data)
            info["momentum"] = layer.momentum
            info["epsilon"] = layer.epsilon
        elseif layer isa Layers.DropoutLayer
            info["rate"] = layer.rate
        end
        
        push!(layer_info, info)
    end
    model_dict["layer_info"] = layer_info
    
    # Guardar
    JLD2.save(filepath, "model", model_dict)
end


mutable struct ProgressCallback <: AbstractCallback
    verbose::Int
    total_batches::Int
    epoch_start_time::Float64
    
    ProgressCallback(verbose::Int=1) = new(verbose, 0, 0.0)
end

function on_epoch_begin(cb::ProgressCallback, epoch::Int, logs::Dict)
    cb.epoch_start_time = time()
end

function on_batch_end(cb::ProgressCallback, batch::Int, logs::Dict)
    if cb.verbose > 1 && batch % max(1, cb.total_batches ÷ 20) == 0
        progress = batch / cb.total_batches
        bar_length = 30
        filled = Int(round(progress * bar_length))
        bar = "█" ^ filled * "░" ^ (bar_length - filled)
        loss = get(logs, :loss, 0.0)
        print("\r  [$bar] $(round(Int, progress*100))% - loss: $(round(loss, digits=4))")
        batch == cb.total_batches && println()
    end
end


end # module Callbacks