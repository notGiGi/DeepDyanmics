# src/Logging.jl
module Logging

using JSON
using Dates
using CUDA
using LinearAlgebra
using Statistics
using ..TensorEngine
using ..NeuralNetwork
export close,TrainingLogger, TensorBoardLogger, ExperimentTracker,
       log_metrics!, flush_logs!, create_experiment, get_hardware_info, log_scalar, log_histogram, save_model_checkpoint, 
        get_git_info, compare_experiments, log_gradients!
import ..Callbacks: AbstractCallback, on_epoch_begin, on_batch_end, on_epoch_end, on_train_end
import Base: close
# ==================================================================
# TrainingLogger - Logger principal con formato JSON Lines
# ==================================================================
mutable struct TrainingLogger
    filepath::String
    log_level::Symbol  # :batch, :epoch
    metrics_history::Dict{String, Vector{Float32}}
    start_time::Float64
    hardware_info::Dict{String, Any}
    buffer::Vector{String}  # Buffer para escritura asíncrona
    buffer_size::Int
    file_handle::Union{IOStream, Nothing}
    async_task::Union{Task, Nothing}
end

function TrainingLogger(filepath::String; 
                       log_level::Symbol=:epoch, 
                       buffer_size::Int=100)
    # Obtener info de hardware
    hardware_info = get_hardware_info()
    
    # Crear directorio si no existe
    dir = dirname(filepath)
    !isempty(dir) && !isdir(dir) && mkpath(dir)
    
    # Abrir archivo
    file_handle = open(filepath, "a")
    
    logger = TrainingLogger(
        filepath,
        log_level,
        Dict{String, Vector{Float32}}(),
        time(),
        hardware_info,
        String[],
        buffer_size,
        file_handle,
        nothing
    )
    
    # Log inicial con info del sistema
    initial_log = Dict(
        "event" => "training_start",
        "timestamp" => Dates.now(),
        "hardware" => hardware_info,
        "julia_version" => string(VERSION)
    )
    write_log(logger, initial_log)
    flush_logs!(logger)
    
    return logger
end

function get_hardware_info()
    info = Dict{String, Any}()
    
    # CPU info
    info["cpu_threads"] = Threads.nthreads()
    info["cpu_model"] = Sys.cpu_info()[1].model
    
    # GPU info
    if CUDA.functional()
        device = CUDA.device()
        info["gpu_available"] = true
        info["gpu_name"] = CUDA.name(device)
        info["gpu_memory_total"] = CUDA.totalmem(device) / 1e9  # GB
        # CAMBIAR ESTA LÍNEA:
        info["cuda_version"] = string(CUDA.runtime_version())  # ← Usar runtime_version()
    else
        info["gpu_available"] = false
    end
    
    # System memory
    info["system_memory_gb"] = Sys.total_memory() / 1e9
    
    return info
end

function write_log(logger::TrainingLogger, data::Dict)
    # Añadir timestamp si no existe
    if !haskey(data, "timestamp")
        data["timestamp"] = Dates.now()
    end
    
    # Convertir a JSON y añadir al buffer
    json_str = JSON.json(data)
    push!(logger.buffer, json_str)
    
    # Escribir si el buffer está lleno
    if length(logger.buffer) >= logger.buffer_size
        flush_logs!(logger)
    end
end

function flush_logs!(logger::TrainingLogger)
    if isempty(logger.buffer)
        return
    end

    # Escritura síncrona
    for line in logger.buffer
        println(logger.file_handle, line)
    end
    empty!(logger.buffer)
    flush(logger.file_handle)
end

function log_metrics!(logger::TrainingLogger, metrics::Dict; 
                     epoch::Int, batch::Union{Int,Nothing}=nothing)
    # Solo loguear según el nivel configurado
    if logger.log_level == :epoch && batch !== nothing
        return
    end
    
    log_data = Dict{String, Any}(
        "event" => "metrics",
        "epoch" => epoch,
        "time_elapsed" => time() - logger.start_time
    )
    
    if batch !== nothing
        log_data["batch"] = batch
    end

    # Añadir métricas
    for (k, v) in metrics
        k == :model && continue  # Skipear explícitamente :model
        if !isa(v, Number)
            if k in [:model, :optimizer, :params]
                continue
            else
                @debug "Métrica no numérica ignorada: $k => $(typeof(v))"
                continue
            end
        end
        log_data[string(k)] = v
        
        # IMPORTANTE: Inicializar el array si no existe
        key = string(k)
        if !haskey(logger.metrics_history, key)
            logger.metrics_history[key] = Float32[]
        end
        push!(logger.metrics_history[key], Float32(v))
    end
    
    # Añadir uso de memoria si GPU disponible
    if CUDA.functional()
        # Usar funciones individuales que sí existen
        free_mem = CUDA.available_memory() / 1e9
        total_mem = CUDA.total_memory() / 1e9
        used_mem = total_mem - free_mem
        
        log_data["gpu_memory_used_gb"] = used_mem
        log_data["gpu_memory_free_gb"] = free_mem
        log_data["gpu_memory_total_gb"] = total_mem
    end
    
    write_log(logger, log_data)
end

function close(logger::TrainingLogger)
    # sincroniza buffer y cierra el stream
    flush_logs!(logger)
    if logger.file_handle !== nothing
        close(logger.file_handle)      # ahora dispatcha a Base.close
        logger.file_handle = nothing
    end
end

function log_gradients!(logger::TrainingLogger, model, epoch::Int)
    params = NeuralNetwork.collect_parameters(model)
    grad_norms = Float32[]
    
    for p in params
        if p.grad !== nothing
            push!(grad_norms, norm(p.grad.data))
        end
    end
    
    if !isempty(grad_norms)
        log_data = Dict(
            "event" => "gradients",
            "epoch" => epoch,
            "mean_grad_norm" => mean(grad_norms),
            "max_grad_norm" => maximum(grad_norms),
            "min_grad_norm" => minimum(grad_norms)
        )
        write_log(logger, log_data)
    end
end

# Callback para integración con el sistema de training
mutable struct LoggingCallback <: AbstractCallback
    logger::TrainingLogger
    log_gradients::Bool
    current_epoch::Int
end

LoggingCallback(logger::TrainingLogger; log_gradients::Bool=false) =
    LoggingCallback(logger, log_gradients, 0)


function on_epoch_begin(cb::LoggingCallback, epoch::Int)
    cb.current_epoch = epoch
end

function on_batch_end(cb::LoggingCallback, batch::Int, logs::Dict)
    if cb.logger.log_level == :batch
        log_metrics!(cb.logger, logs, epoch=cb.current_epoch, batch=batch)
    end
end

function on_epoch_end(cb::LoggingCallback, epoch::Int, logs::Dict)
    #@info "🚧 on_epoch_end llamado para epoch=$epoch"
    log_metrics!(cb.logger, logs, epoch=epoch)
    
    if cb.log_gradients && haskey(logs, :model)
        log_gradients!(cb.logger, logs[:model], epoch)
    end
    
    # Flush periódicamente
    if epoch % 10 == 0
        flush_logs!(cb.logger)
    end
end

function on_train_end(cb::LoggingCallback, logs::Dict)
    #@info "🚧 on_train_end llamado, finalizando logs"
    final_log = Dict(
        "event"         => "training_end",
        "total_time"    => time() - cb.logger.start_time,
        "final_metrics" => Dict(k => v[end] for (k,v) in cb.logger.metrics_history if !isempty(v))
    )
    write_log(cb.logger, final_log)
    flush_logs!(cb.logger)
    close(cb.logger)
end


# ==================================================================
# TensorBoardLogger - Logging compatible con TensorBoard
# ==================================================================
# Estructura simplificada para TensorBoard (formato básico)
struct TensorBoardWriter
    scalars_dir::String
    histograms_dir::String
    images_dir::String
end


mutable struct TensorBoardLogger
    log_dir::String
    writer::TensorBoardWriter
    step::Int
end



function TensorBoardLogger(log_dir::String)
    # Crear estructura de directorios
    scalars_dir = joinpath(log_dir, "scalars")
    histograms_dir = joinpath(log_dir, "histograms") 
    images_dir = joinpath(log_dir, "images")
    
    for dir in [scalars_dir, histograms_dir, images_dir]
        !isdir(dir) && mkpath(dir)
    end
    
    writer = TensorBoardWriter(scalars_dir, histograms_dir, images_dir)
    return TensorBoardLogger(log_dir, writer, 0)
end

function log_scalar(tb::TensorBoardLogger, tag::String, value::Float32, step::Int)
    # Formato simplificado compatible con TensorBoard
    safe_tag = replace(tag, "/" => "_")
    filepath = joinpath(tb.writer.scalars_dir, "$(safe_tag).json")

    
    data = Dict(
        "step" => step,
        "value" => value,
        "wall_time" => time()
    )
    
    open(filepath, "a") do f
        println(f, JSON.json(data))
    end
end

function log_histogram(
    tb::TensorBoardLogger,
    tag::String,
    values::AbstractArray{<:Number},
    step::Int
)
    if values isa CUDA.CuArray
        values = Array(values)  # Mover a CPU
    end    
    # 1) Split tag en subdirectorios + filename
    parts    = split(tag, "/")
    subdirs  = parts[1:end-1]
    filename = parts[end] * ".json"

    # 2) Crear ruta de directorio
    base_dir = tb.writer.histograms_dir
    dirpath  = isempty(subdirs) ? base_dir : joinpath(base_dir, subdirs...)
    !isdir(dirpath) && mkpath(dirpath)

    # 3) Archivo destino
    filepath = joinpath(dirpath, filename)

    # 4) Aplanar el array para vector/matriz/tensor
    flat_vals = vec(values)

    # 5) Estadísticas básicas en un Dict{String, Any}
    hist_data = Dict{String, Any}(
        "step"      => step,
        "wall_time" => time(),
        "min"       => minimum(flat_vals),
        "max"       => maximum(flat_vals),
        "mean"      => mean(flat_vals),
        "std"       => std(flat_vals),
        "count"     => length(flat_vals)
    )

    # 6) Bins y counts
    n_bins = min(30, length(unique(flat_vals)))
    edges  = range(minimum(flat_vals), maximum(flat_vals), length = n_bins + 1)
    counts = zeros(Int, length(edges) - 1)
    for v in flat_vals
        for i in eachindex(counts)
            if (edges[i] <= v < edges[i+1]) || (i == lastindex(counts) && v == edges[end])
                counts[i] += 1
                break
            end
        end
    end
    hist_data["bins"]   = collect(edges)
    hist_data["counts"] = counts

    # 7) Escritura única
    open(filepath, "a") do io
        println(io, JSON.json(hist_data))
    end

    return hist_data
end





# TensorBoard Callback
mutable struct TensorBoardCallback <: AbstractCallback
    tb_logger::TensorBoardLogger
    log_weights::Bool
    log_gradients::Bool
    current_epoch::Int
end

TensorBoardCallback(log_dir::String; log_weights=true, log_gradients=true) =
    TensorBoardCallback(TensorBoardLogger(log_dir), log_weights, log_gradients, 0)

function on_epoch_begin(cb::TensorBoardCallback, epoch::Int)
    cb.current_epoch = epoch
end

function on_epoch_end(cb::TensorBoardCallback, epoch::Int, logs::Dict)
    cb.tb_logger.step = epoch
    
    # Log métricas escalares
    for (k, v) in logs
        if isa(v, Number) && k != :model
            log_scalar(cb.tb_logger, string(k), Float32(v), epoch)
        end
    end
    
    # Log pesos y gradientes si está habilitado
    if haskey(logs, :model)
        model = logs[:model]
        params = NeuralNetwork.collect_parameters(model)
        
        for (i, p) in enumerate(params)
            if cb.log_weights
                # Asegurar que está en CPU antes de log_histogram
                weight_data = p.data isa CUDA.CuArray ? Array(p.data) : p.data
                log_histogram(cb.tb_logger, "weights/param_$i", weight_data, epoch)
            end
            
            if cb.log_gradients && p.grad !== nothing
                # Asegurar que está en CPU
                grad_data = p.grad.data isa CUDA.CuArray ? Array(p.grad.data) : p.grad.data
                log_histogram(cb.tb_logger, "gradients/param_$i", grad_data, epoch)
            end
        end
    end
end

# ==================================================================
# ExperimentTracker - Sistema de tracking de experimentos
# ==================================================================
mutable struct ExperimentTracker
    experiment_id::String
    base_dir::String
    config::Dict{String, Any}
    start_time::DateTime
    git_info::Dict{String, String}
end

function create_experiment(base_dir::String, config::Dict{String, Any})
    # Generar ID único
    experiment_id = string(Dates.format(now(), "yyyymmdd_HHMMSS")) * "_" * 
                   string(rand(UInt16))
    
    # Crear directorio del experimento
    exp_dir = joinpath(base_dir, experiment_id)
    mkpath(exp_dir)
    
    # Obtener info de git si está disponible
    git_info = get_git_info()
    
    # Guardar configuración
    full_config = merge(config, Dict(
        "experiment_id" => experiment_id,
        "start_time" => now(),
        "git_info" => git_info,
        "julia_version" => string(VERSION),
        "hardware_info" => get_hardware_info()
    ))
    
    # Guardar config
    config_path = joinpath(exp_dir, "config.json")
    open(config_path, "w") do f
        JSON.print(f, full_config, 2)
    end
    
    tracker = ExperimentTracker(
        experiment_id,
        exp_dir,
        full_config,
        now(),
        git_info
    )
    
    return tracker
end

function get_git_info()
    info = Dict{String, String}()
    
    try
        # Obtener commit hash
        info["commit"] = strip(read(`git rev-parse HEAD`, String))
        
        # Verificar si hay cambios no commiteados
        status = read(`git status --porcelain`, String)
        info["has_changes"] = !isempty(status)
        
        # Branch actual
        info["branch"] = strip(read(`git rev-parse --abbrev-ref HEAD`, String))
    catch
        info["available"] = "false"
    end
    
    return info
end

function save_model_checkpoint(tracker::ExperimentTracker, model, epoch::Int, metrics::Dict)
    checkpoint_dir = joinpath(tracker.base_dir, "checkpoints")
    !isdir(checkpoint_dir) && mkpath(checkpoint_dir)
    
    checkpoint_path = joinpath(checkpoint_dir, "checkpoint_epoch_$(epoch).jld2")
    
    # Guardar usando ModelSaver si está disponible
    checkpoint_data = Dict(
        "epoch" => epoch,
        "metrics" => metrics,
        "timestamp" => now()
    )
    
    # TODO: Integrar con ModelSaver cuando esté disponible
    # Por ahora guardamos la metadata
    metadata_path = joinpath(checkpoint_dir, "checkpoint_epoch_$(epoch)_meta.json")
    open(metadata_path, "w") do f
        JSON.print(f, checkpoint_data, 2)
    end
end

# Utilidad para comparar experimentos
function compare_experiments(exp_ids::Vector{String}, base_dir::String)
    comparisons = Dict{String, Any}[]
    
    for exp_id in exp_ids
        config_path = joinpath(base_dir, exp_id, "config.json")
        if isfile(config_path)
            config = JSON.parsefile(config_path)
            
            # Cargar métricas finales si existen
            logs_path = joinpath(base_dir, exp_id, "training.jsonl")
            final_metrics = Dict{String, Any}()
            
            if isfile(logs_path)
                # Leer última línea con métricas
                for line in eachline(logs_path)
                    data = JSON.parse(line)
                    if get(data, "event", "") == "training_end"
                        final_metrics = get(data, "final_metrics", Dict())
                        break
                    end
                end
            end
            
            push!(comparisons, Dict(
                "experiment_id" => exp_id,
                "config" => config,
                "final_metrics" => final_metrics
            ))
        end
    end
    
    return comparisons
end

# Integración con fit!
function setup_logging(experiment_name::String, config::Dict;
                      log_dir::String="experiments",
                      use_tensorboard::Bool=false,
                      log_gradients::Bool=false)
    
    # Crear experimento
    tracker = create_experiment(log_dir, config)
    
    # Logger principal
    log_path = joinpath(tracker.base_dir, "training.jsonl")
    logger = TrainingLogger(log_path, log_level=:epoch)
    
    callbacks = AbstractCallback[LoggingCallback(logger, log_gradients=log_gradients)]
    
    # TensorBoard si está habilitado
    if use_tensorboard
        tb_callback = TensorBoardCallback(
            joinpath(tracker.base_dir, "tensorboard"),
            log_weights=true,
            log_gradients=log_gradients
        )
        push!(callbacks, tb_callback)
    end
    
    return tracker, callbacks
end

# Para compatibilidad con StatsBase si está disponible
try
    using StatsBase: Histogram, fit
catch
    # Implementación básica de histograma
    struct Histogram
        edges::Vector{Float64}
        weights::Vector{Int}
    end
    
    function fit(::Type{Histogram}, values::Vector, edges::Vector)
        weights = zeros(Int, length(edges)-1)
        for v in values
            for i in 1:length(edges)-1
                if edges[i] <= v < edges[i+1]
                    weights[i] += 1
                    break
                end
            end
        end
        return Histogram(collect(edges), weights)
    end
end

end # module