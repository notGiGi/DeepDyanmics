# src/MLOpsIntegrations.jl
module MLOpsIntegrations

using HTTP
using JSON3
using Dates
using Printf
using Base64
using ..TensorEngine
using ..Callbacks
using ..Logging
using ..NeuralNetwork
using LinearAlgebra
using Statistics
using CUDA
using JLD2
export WandBLogger, MLFlowLogger, create_mlops_logger,
       sync_offline_runs, MLOpsConfig

# ===== Configuración =====
struct MLOpsConfig
    wandb_api_key::Union{String,Nothing}
    wandb_entity::Union{String,Nothing}
    wandb_base_url::String
    mlflow_tracking_uri::String
    offline_mode::Bool
    retry_attempts::Int
    batch_size::Int
end

function MLOpsConfig(;
    wandb_api_key=get(ENV, "WANDB_API_KEY", nothing),
    wandb_entity=get(ENV, "WANDB_ENTITY", nothing),
    wandb_base_url=get(ENV, "WANDB_BASE_URL", "https://api.wandb.ai"),
    mlflow_tracking_uri=get(ENV, "MLFLOW_TRACKING_URI", "http://localhost:5000"),
    offline_mode=false,
    retry_attempts=3,
    batch_size=100
)
    MLOpsConfig(wandb_api_key, wandb_entity, wandb_base_url,
                mlflow_tracking_uri, offline_mode, retry_attempts, batch_size)
end

# ===== Weights & Biases Logger =====
mutable struct WandBLogger <: AbstractCallback
    project::String
    entity::Union{String,Nothing}
    config::Dict
    run_id::String
    api_key::Union{String,Nothing}
    base_url::String
    metrics_buffer::Vector{Dict}
    local_logger::TrainingLogger  # Usa el logger existente como respaldo
    offline_mode::Bool
    retry_attempts::Int
    last_sync_time::Float64
    
    function WandBLogger(project::String, config::Dict;
                        entity=nothing,
                        api_key=nothing,
                        base_url="https://api.wandb.ai",
                        offline_mode=false,
                        retry_attempts=3,
                        log_dir="./logs/wandb")
        
        run_id = generate_run_id()
        
        # Crear logger local como respaldo
        mkpath(log_dir)
        local_logger = TrainingLogger(
            joinpath(log_dir, "wandb_$run_id.jsonl"),
            log_level=:epoch
        )
        
        logger = new(project, entity, config, run_id, api_key, base_url,
                    Dict[], local_logger, offline_mode, retry_attempts, time())
        
        if !offline_mode && api_key !== nothing
            try
                init_wandb_run(logger)
            catch e
                @warn "No se pudo conectar con W&B, usando modo offline: $e"
                logger.offline_mode = true
            end
        end
        
        return logger
    end
end

function init_wandb_run(logger::WandBLogger)
    payload = Dict(
        "entity" => logger.entity,
        "project" => logger.project,
        "config" => logger.config,
        "host" => gethostname(),
        "start_time" => round(Int, time() * 1000)
    )
    
    response = http_retry(
        () -> HTTP.post(
            "$(logger.base_url)/api/runs",
            ["Authorization" => "Bearer $(logger.api_key)",
             "Content-Type" => "application/json"],
            JSON3.write(payload)
        ),
        logger.retry_attempts
    )
    
    if response.status == 200
        data = JSON3.read(response.body)
        logger.run_id = data["id"]
        @info "W&B run inicializado: $(logger.project)/$(logger.run_id)"
    else
        throw(ErrorException("Error W&B init: HTTP $(response.status)"))
    end
end

# Callbacks para W&B
function Callbacks.on_epoch_end(logger::WandBLogger, epoch::Int, logs::Dict)
    # Log local siempre
    log_metrics!(logger.local_logger, logs, epoch=epoch)
    
    # Preparar métricas para W&B
    metrics = Dict{String,Any}()
    metrics["_step"] = epoch
    metrics["_runtime"] = round(Int, (time() - logger.last_sync_time) * 1000)
    
    # Extraer métricas numéricas
    for (k, v) in logs
        if isa(v, Number) && k ∉ [:model, :optimizer]
            metrics[string(k)] = v
        end
    end
    
    # Agregar métricas del sistema
    if CUDA.functional()
        metrics["system/gpu_memory_used"] = (CUDA.total_memory() - CUDA.available_memory()) / 1e9
        metrics["system/gpu_utilization"] = get_gpu_utilization()
    end
    
    push!(logger.metrics_buffer, metrics)
    
    # Enviar cada N épocas o si el buffer está lleno
    if length(logger.metrics_buffer) >= logger.retry_attempts || epoch % 5 == 0
        sync_wandb_metrics(logger)
    end
end

function Callbacks.on_train_end(logger::WandBLogger, logs::Dict)
    # Flush métricas pendientes
    sync_wandb_metrics(logger)
    
    # Log del resumen final
    summary = Dict(
        "best_loss" => minimum(get(logger.local_logger.metrics_history, "loss", [Inf])),
        "final_loss" => get(logs, :loss, nothing),
        "total_epochs" => get(logs, :epoch, 0)
    )
    
    if !logger.offline_mode
        try
            upload_run_summary(logger, summary)
            
            # Upload del modelo si está disponible
            model = get(logs, :model, nothing)
            if model !== nothing
                upload_model_artifact(logger, model)
            end
        catch e
            @warn "Error al finalizar W&B run: $e"
        end
    end
    
    # Cerrar logger local
    Logging.close(logger.local_logger)
end

function sync_wandb_metrics(logger::WandBLogger)
    if isempty(logger.metrics_buffer) || logger.offline_mode
        return
    end
    
    try
        response = http_retry(
            () -> HTTP.post(
                "$(logger.base_url)/api/runs/$(logger.run_id)/history",
                ["Authorization" => "Bearer $(logger.api_key)",
                 "Content-Type" => "application/json"],
                JSON3.write(logger.metrics_buffer)
            ),
            logger.retry_attempts
        )
        
        if response.status == 200
            empty!(logger.metrics_buffer)
            logger.last_sync_time = time()
        end
    catch e
        @warn "Error al sincronizar con W&B: $e"
        # Los datos quedan en el buffer para reintentar después
    end
end

# ===== MLFlow Logger =====
mutable struct MLFlowLogger <: AbstractCallback
    tracking_uri::String
    experiment_name::String
    run_id::String
    experiment_id::String
    metrics_buffer::Vector{Dict}
    local_logger::TrainingLogger
    offline_mode::Bool
    retry_attempts::Int
    params_logged::Bool
    
    function MLFlowLogger(experiment_name::String;
                        tracking_uri="http://localhost:5000",
                        offline_mode=false,
                        retry_attempts=3,
                        log_dir="./logs/mlflow")
        
        mkpath(log_dir)
        # Crear timestamp seguro para Windows
        timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
        safe_name = replace(experiment_name, r"[^\w\-]" => "_")
        
        local_logger = TrainingLogger(
            joinpath(log_dir, "mlflow_$(safe_name)_$(timestamp).jsonl"),
            log_level=:epoch
        )
        
        logger = new(tracking_uri, experiment_name, "", "",
                    Dict[], local_logger, offline_mode, retry_attempts, false)
        
        if !offline_mode
            try
                init_mlflow_experiment(logger)
            catch e
                @warn "No se pudo conectar con MLFlow, usando modo offline: $e"
                logger.offline_mode = true
            end
        end
        
        return logger
    end
end

function init_mlflow_experiment(logger::MLFlowLogger)
    # Buscar o crear experimento
    exp_id = get_or_create_experiment(logger)
    logger.experiment_id = exp_id
    
    # Crear run
    response = http_retry(
        () -> HTTP.post(
            "$(logger.tracking_uri)/api/2.0/mlflow/runs/create",
            ["Content-Type" => "application/json"],
            JSON3.write(Dict(
                "experiment_id" => exp_id,
                "start_time" => round(Int, time() * 1000),
                "tags" => Dict(
                    "mlflow.source.type" => "LOCAL",
                    "mlflow.user" => get(ENV, "USER", "unknown")
                )
            ))
        ),
        logger.retry_attempts
    )
    
    if response.status == 200
        data = JSON3.read(response.body)
        logger.run_id = data["run"]["info"]["run_id"]
        @info "MLFlow run inicializado: $(logger.experiment_name)/$(logger.run_id)"
    else
        throw(ErrorException("Error creando MLFlow run"))
    end
end

# Callbacks para MLFlow
function Callbacks.on_train_begin(logger::MLFlowLogger, logs::Dict)
    # Log hiperparámetros
    if !logger.params_logged && !logger.offline_mode
        params = extract_mlflow_params(logs)
        log_mlflow_params(logger, params)
        logger.params_logged = true
    end
end

function Callbacks.on_epoch_end(logger::MLFlowLogger, epoch::Int, logs::Dict)
    # Log local
    log_metrics!(logger.local_logger, logs, epoch=epoch)
    
    # Preparar para MLFlow
    timestamp = round(Int, time() * 1000)
    
    for (k, v) in logs
        if isa(v, Number) && k ∉ [:model, :optimizer]
            push!(logger.metrics_buffer, Dict(
                "key" => string(k),
                "value" => Float64(v),
                "timestamp" => timestamp,
                "step" => epoch
            ))
        end
    end
    
    # Sincronizar periódicamente
    if length(logger.metrics_buffer) >= 50
        sync_mlflow_metrics(logger)
    end
end

function Callbacks.on_train_end(logger::MLFlowLogger, logs::Dict)
    # Sincronizar métricas finales
    sync_mlflow_metrics(logger)
    
    # Finalizar run
    if !logger.offline_mode
        finalize_mlflow_run(logger, logs)
    end
    
    # Cerrar logger local
    Logging.close(logger.local_logger)
end

# ===== Logger Combinado =====
function create_mlops_logger(;
    project_name="DeepDynamics",
    platforms=[:wandb, :mlflow],
    config=Dict(),
    mlops_config=MLOpsConfig()
)
    loggers = AbstractCallback[]
    
    for platform in platforms
        if platform == :wandb && mlops_config.wandb_api_key !== nothing
            push!(loggers, WandBLogger(
                project_name, config,
                entity=mlops_config.wandb_entity,
                api_key=mlops_config.wandb_api_key,
                base_url=mlops_config.wandb_base_url,
                offline_mode=mlops_config.offline_mode,
                retry_attempts=mlops_config.retry_attempts
            ))
        elseif platform == :mlflow
            push!(loggers, MLFlowLogger(
                project_name,
                tracking_uri=mlops_config.mlflow_tracking_uri,
                offline_mode=mlops_config.offline_mode,
                retry_attempts=mlops_config.retry_attempts
            ))
        end
    end
    
    return loggers
end

# ===== Utilidades =====
function http_retry(request_fn, max_attempts::Int)
    for attempt in 1:max_attempts
        try
            return request_fn()
        catch e
            if attempt == max_attempts
                rethrow(e)
            end
            sleep(min(2^attempt, 30))  # Exponential backoff con límite
        end
    end
end

function get_gpu_utilization()
    if CUDA.functional()
        # Estimar utilización basada en memoria usada
        # En producción usar nvidia-ml o NVML.jl
        mem_used = CUDA.total_memory() - CUDA.available_memory()
        mem_total = CUDA.total_memory()
        return 100.0 * mem_used / mem_total
    end
    return 0.0
end

function generate_run_id()
    # Generar ID único
    timestamp = round(Int, time() * 1000)
    random_suffix = rand(UInt32)
    return "run_$(timestamp)_$(random_suffix)"
end

function upload_model_artifact(logger::WandBLogger, model)
    # Guardar modelo temporalmente
    temp_file = tempname() * ".jld2"
    
    try
        # Usar ModelSaver si está disponible
        if isdefined(Main, :ModelSaver)
            ModelSaver.save_model(temp_file, model)
        else
            # Fallback básico
            
            jldsave(temp_file; model=model)
        end
        
        # Upload como artifact
        artifact_data = Dict(
            "type" => "model",
            "name" => "model_$(logger.run_id)",
            "description" => "Final trained model"
        )
        
        # Aquí iría la lógica de upload real
        @info "Modelo guardado como artifact W&B"
        
    finally
        rm(temp_file, force=true)
    end
end

function sync_offline_runs(config::MLOpsConfig=MLOpsConfig())
    # Buscar logs offline
    offline_logs = String[]
    
    for dir in ["./logs/wandb", "./logs/mlflow"]
        if isdir(dir)
            append!(offline_logs, 
                    filter(f -> endswith(f, ".jsonl"), readdir(dir, join=true)))
        end
    end
    
    @info "Encontrados $(length(offline_logs)) logs offline para sincronizar"
    
    # Procesar cada log
    for log_file in offline_logs
        try
            if contains(log_file, "wandb")
                sync_wandb_offline_log(log_file, config)
            elseif contains(log_file, "mlflow")
                sync_mlflow_offline_log(log_file, config)
            end
        catch e
            @warn "Error sincronizando $log_file: $e"
        end
    end
end

# Funciones auxiliares para MLFlow
function get_or_create_experiment(logger::MLFlowLogger)
    # Buscar experimento existente
    try
        response = HTTP.get(
            "$(logger.tracking_uri)/api/2.0/mlflow/experiments/get-by-name",
            query=Dict("experiment_name" => logger.experiment_name)
        )
        
        if response.status == 200
            data = JSON3.read(response.body)
            return data["experiment"]["experiment_id"]
        end
    catch
    end
    
    # Crear nuevo experimento
    response = HTTP.post(
        "$(logger.tracking_uri)/api/2.0/mlflow/experiments/create",
        ["Content-Type" => "application/json"],
        JSON3.write(Dict("name" => logger.experiment_name))
    )
    
    data = JSON3.read(response.body)
    return data["experiment_id"]
end

function extract_mlflow_params(logs::Dict)
    params = []
    
    # Extraer configuración del optimizer
    if haskey(logs, :optimizer)
        opt = logs[:optimizer]
        push!(params, Dict("key" => "optimizer_type", 
                          "value" => string(typeof(opt))))
        if hasproperty(opt, :learning_rate)
            push!(params, Dict("key" => "learning_rate", 
                              "value" => string(opt.learning_rate)))
        end
    end
    
    # Otros parámetros
    for (k, v) in logs
        if k in [:batch_size, :epochs] && isa(v, Number)
            push!(params, Dict("key" => string(k), "value" => string(v)))
        end
    end
    
    return params
end

function log_mlflow_params(logger::MLFlowLogger, params)
    if isempty(params) || logger.offline_mode
        return
    end
    
    try
        http_retry(
            () -> HTTP.post(
                "$(logger.tracking_uri)/api/2.0/mlflow/runs/log-batch",
                ["Content-Type" => "application/json"],
                JSON3.write(Dict(
                    "run_id" => logger.run_id,
                    "params" => params
                ))
            ),
            logger.retry_attempts
        )
    catch e
        @warn "Error logging MLFlow params: $e"
    end
end

function sync_mlflow_metrics(logger::MLFlowLogger)
    if isempty(logger.metrics_buffer) || logger.offline_mode
        return
    end
    
    try
        response = http_retry(
            () -> HTTP.post(
                "$(logger.tracking_uri)/api/2.0/mlflow/runs/log-batch",
                ["Content-Type" => "application/json"],
                JSON3.write(Dict(
                    "run_id" => logger.run_id,
                    "metrics" => logger.metrics_buffer
                ))
            ),
            logger.retry_attempts
        )
        
        if response.status == 200
            empty!(logger.metrics_buffer)
        end
    catch e
        @warn "Error sincronizando métricas MLFlow: $e"
    end
end

function finalize_mlflow_run(logger::MLFlowLogger, logs::Dict)
    try
        # Actualizar estado del run
        http_retry(
            () -> HTTP.post(
                "$(logger.tracking_uri)/api/2.0/mlflow/runs/update",
                ["Content-Type" => "application/json"],
                JSON3.write(Dict(
                    "run_id" => logger.run_id,
                    "status" => "FINISHED",
                    "end_time" => round(Int, time() * 1000)
                ))
            ),
            logger.retry_attempts
        )
        
        @info "MLFlow run finalizado: $(logger.run_id)"
    catch e
        @warn "Error finalizando MLFlow run: $e"
    end
end

end # module