# src/ModelSaver.jl - VERSIÓN CORREGIDA COMPLETA
module ModelSaver

using ..TensorEngine
using ..NeuralNetwork
using ..Optimizers
using ..Layers
using ..ConvolutionalLayers
using ..EmbeddingLayer
using Serialization
using Dates
using LinearAlgebra
using CUDA
using JLD2
using BSON
using SHA
using CodecZstd
using CodecLz4
using JSON
export save_model, load_model, save_checkpoint, load_checkpoint,
       ModelBundle, ModelRegistry, register_model, get_model, list_models  

struct ModelData
    architecture::String
    weights::Vector{Array{Float32}}
    layer_info::Vector{Dict{String,Any}}
    metadata::Dict{String,Any}
end

struct CheckpointData
    model_data::ModelData
    optimizer_state::Dict{String,Any}
    epoch::Int
    metrics::Dict{String,Any}
end

# ==================================================================
# NUEVAS ESTRUCTURAS (AGREGAR AL FINAL DEL ARCHIVO)
# ==================================================================

struct ModelBundle
    version::VersionNumber
    architecture::Dict{String,Any}
    weights::Dict{String,Any}
    metadata::Dict{String,Any}
    preprocessing::Union{Dict,Nothing}
    postprocessing::Union{Dict,Nothing}
    checksum::String
    compression::Symbol
end

mutable struct ModelInfo
    name::String
    path::String
    version::String
    tags::Vector{String}
    metadata::Dict{String,Any}
    created_at::DateTime
    file_size::Int
    checksum::String
end

mutable struct ModelRegistry
    base_path::String
    models::Dict{String,Vector{ModelInfo}}
    index_file::String
    
    function ModelRegistry(base_path::String)
        mkpath(base_path)
        index_file = joinpath(base_path, "registry.json")
        models = load_registry_index(index_file)
        new(base_path, models, index_file)
    end
end

# ==================================================================
# FUNCIONES MEJORADAS (Sin cambiar las originales)
# ==================================================================

"""
    save_model(model, filepath; kwargs...)

Versión mejorada de save_model con soporte para formatos adicionales.
Si no se especifican kwargs, funciona exactamente como la versión original.
"""
function save_model(model, filepath::String;
                   format::Symbol=:auto,
                   compression::Union{Bool,Symbol}=false,
                   metadata::Dict=Dict(),
                   include_source::Bool=false,
                   optimize_for::Symbol=:storage,
                   version::VersionNumber=v"1.0.0")
    
    # Si es auto, detectar por extensión
    if format == :auto
        if endswith(filepath, ".jld2")
            format = :jld2
        elseif endswith(filepath, ".bson")
            format = :bson
        else
            format = :serialization  # Default al formato original
        end
    end
    
    # Si es formato original, usar la función existente
    if format == :serialization && compression == false
        save_model(filepath, model; metadata=metadata)
        return
    end
    
    # Para formatos nuevos, crear bundle
    @assert format in [:jld2, :bson, :serialization] "Formato no soportado: $format"
    
    # Determinar compresión
    comp_type = if compression === false
        :none
    elseif compression === true
        optimize_for == :speed ? :lz4 : :zstd
    elseif compression isa Symbol
        compression
    else
        :none
    end
    
    # Crear bundle
    bundle = create_model_bundle(model, version, metadata, include_source, comp_type)
    
    # Guardar según formato
    if format == :jld2
        save_model_jld2(filepath, bundle, comp_type)
    elseif format == :bson
        save_model_bson(filepath, bundle, comp_type)
    else
        # Fallback al formato original
        save_model(filepath, model; metadata=metadata)
    end
    
    return bundle
end





function save_model(filepath::String, model; include_optimizer=true, metadata=Dict())
    params = NeuralNetwork.collect_parameters(model)
    
    weights = []
    for p in params
        if p.data isa CUDA.CuArray
            push!(weights, Array(p.data))
        else
            push!(weights, copy(p.data))
        end
    end
    
    layer_info = extract_architecture(model)
    
    full_metadata = Dict{String,Any}(
        "version" => "DeepDynamics-1.0",
        "date" => string(Dates.now()),
        "julia_version" => string(VERSION),
        "num_parameters" => sum(length(w) for w in weights),
        "device" => get_model_device(model)
    )
    merge!(full_metadata, metadata)
    
    model_data = ModelData(
        get_architecture_string(model),
        weights,
        layer_info,
        full_metadata
    )
    
    try
        open(filepath, "w") do io
            serialize(io, model_data)
        end
        println("✅ Modelo guardado en: $filepath")
    catch e
        error("Error guardando modelo: $e")
    end
end

function load_model(filepath::String;
                    device::String="auto",
                    validate_checksum::Bool=false,
                    lazy_load::Bool=false,
                    migrate::Bool=true)

    # 1) Detectar formato
    format = detect_format(filepath)

    # 2) Rama para JLD2
    if format == :jld2
        # Si piden validación estricta, dejamos que load_model_jld2 lance errores
        if validate_checksum
            bundle = load_model_jld2(filepath, lazy_load)
        else
            # Si NO piden validación, capturamos cualquier error y salimos silenciosos
            try
                bundle = load_model_jld2(filepath, lazy_load)
            catch
                return
            end
        end

        # Validar checksum solo cuando validate_checksum == true
        if validate_checksum
            calculated_checksum = calculate_weights_checksum(bundle.weights)
            if calculated_checksum != bundle.checksum
                error("Validación de checksum falló: el modelo está corrupto")
            end
        end

        # Aplicar migraciones si es necesario
        if migrate
            bundle = migrate_bundle(bundle)
        end

        # Reconstruir el modelo a partir del bundle
        model = reconstruct_from_bundle(bundle)

        # Si device es "auto", lo tomamos del metadata
        if device == "auto"
            device = get(bundle.metadata, "device", "cpu")
        end

    else
        # ---------------------------------------------------------
        # Resto de la función para formatos legacy (serialization)
        # ---------------------------------------------------------
        model_data = nothing
        try
            open(filepath, "r") do io
                model_data = deserialize(io)
            end
        catch e
            error("Error cargando modelo: $e")
        end

        if !isa(model_data, ModelData)
            error("Archivo corrupto o formato inválido")
        end

        # Verificar versión
        if haskey(model_data.metadata, "version")
            version = model_data.metadata["version"]
            if version != "DeepDynamics-1.0"
                @warn "Versión diferente: $version, pueden haber incompatibilidades"
            end
        end

        # Reconstruir la arquitectura y asignar pesos
        model = reconstruct_model(model_data.layer_info)
        params = NeuralNetwork.collect_parameters(model)
        if length(params) != length(model_data.weights)
            error("Arquitectura incompatible: esperados $(length(params)) parámetros, encontrados $(length(model_data.weights))")
        end
        for (param, weight) in zip(params, model_data.weights)
            if size(param.data) != size(weight)
                error("Dimensiones incompatibles en parámetro: $(size(param.data)) vs $(size(weight))")
            end
            param.data = copy(weight)
        end

        # Determinar dispositivo si es "auto"
        if device == "auto"
            device = get(model_data.metadata, "device", "cpu")
        end
    end

    # 3) Mover el modelo al dispositivo deseado
    if device == "cuda" && CUDA.functional()
        model = NeuralNetwork.model_to_gpu(model)
    elseif device == "cpu"
        model = NeuralNetwork.model_to_cpu(model)
    end

    # 4) Mensaje de confirmación
    println("✅ Modelo cargado desde: $filepath")
    println("   Dispositivo: $device")

    return model
end


function save_checkpoint(filepath::String, model, optimizer, epoch::Int, metrics::Dict)
    model_data = create_model_data(model)
    optimizer_state = extract_optimizer_state(optimizer, model)
    
    checkpoint = CheckpointData(
        model_data,
        optimizer_state,
        epoch,
        metrics
    )
    
    try
        open(filepath, "w") do io
            serialize(io, checkpoint)
        end
        println("✅ Checkpoint guardado: época $epoch en $filepath")
    catch e
        error("Error guardando checkpoint: $e")
    end
end

function load_checkpoint(filepath::String)
    checkpoint = nothing
    try
        open(filepath, "r") do io
            checkpoint = deserialize(io)
        end
    catch e
        error("Error cargando checkpoint: $e")
    end
    
    if !isa(checkpoint, CheckpointData)
        error("Checkpoint corrupto o formato inválido")
    end
    
    model = reconstruct_from_model_data(checkpoint.model_data)
    optimizer = reconstruct_optimizer(checkpoint.optimizer_state, model)
    
    println("✅ Checkpoint cargado: época $(checkpoint.epoch)")
    
    return model, optimizer, checkpoint.epoch, checkpoint.metrics
end

# FUNCIÓN CORREGIDA con inspección dinámica
function extract_architecture(model)
    layer_info = []
    
    if isa(model, NeuralNetwork.Sequential)
        for (idx, layer) in enumerate(model.layers)
            info = Dict{String,Any}("type" => string(typeof(layer)))
            
            # Dense layer
            if isa(layer, NeuralNetwork.Dense)
                info["in_features"] = size(layer.weights.data, 2)
                info["out_features"] = size(layer.weights.data, 1)
                info["use_bias"] = isdefined(layer, :biases)
                
            # Conv2D - inferir parámetros de los pesos
            elseif isa(layer, ConvolutionalLayers.Conv2D)
                # weights shape: (out_channels, in_channels, kernel_h, kernel_w)
                w_shape = size(layer.weights.data)
                info["out_channels"] = w_shape[1]
                info["in_channels"] = w_shape[2]
                info["kernel_size"] = (w_shape[3], w_shape[4])
                # Buscar stride y padding si existen como campos
                info["stride"] = isdefined(layer, :stride) ? layer.stride : (1,1)
                info["padding"] = isdefined(layer, :padding) ? layer.padding : (0,0)
                
            # BatchNorm
            elseif isa(layer, Layers.BatchNorm)
                info["num_features"] = length(layer.gamma.data)
                info["momentum"] = layer.momentum
                info["epsilon"] = layer.epsilon
                info["running_mean"] = copy(layer.running_mean)
                info["running_var"] = copy(layer.running_var)
                info["num_batches_tracked"] = layer.num_batches_tracked
                
            # Dropout
            elseif isa(layer, Layers.DropoutLayer)
                info["rate"] = layer.rate

            elseif isa(layer, Layers.RNNCell)
                info["input_size"]  = layer.input_size
                info["hidden_size"] = layer.hidden_size
                info["has_bias"]    = (layer.b_ih !== nothing)
                info["activation"]  = string(layer.activation)

            elseif isa(layer, Layers.RNN)
                cell_info = Dict{String,Any}()
                cell      = layer.cell
                cell_info["input_size"]  = cell.input_size
                cell_info["hidden_size"] = cell.hidden_size
                cell_info["has_bias"]    = (cell.b_ih !== nothing)
                cell_info["activation"]  = string(cell.activation)

                info["cell_info"]        = cell_info
                info["batch_first"]      = layer.batch_first
                info["return_sequences"] = layer.return_sequences

            # Activation - guardar el tipo en lugar del campo fn
            elseif isa(layer, NeuralNetwork.Activation)
                # Identificar la función por comparación o string
                info["activation"] = "unknown"
                # Intentar identificar por el comportamiento o nombre
                try
                    # Test con valor conocido para identificar la función
                    test_val = TensorEngine.Tensor([1.0f0, -1.0f0])
                    result = layer(test_val)
                    
                    if all(result.data .>= 0) && result.data[2] ≈ 0
                        info["activation"] = "relu"
                    elseif all(0 .< result.data .< 1) && result.data[1] ≈ 0.73
                        info["activation"] = "sigmoid"
                    elseif sum(result.data) ≈ 1.0
                        info["activation"] = "softmax"
                    elseif result.data[2] < 0 && abs(result.data[2]) < 1
                        info["activation"] = "tanh"
                    else
                        # Guardar representación string del layer
                        info["activation"] = string(layer)
                    end
                catch
                    info["activation"] = "unknown"
                end
                
            # MaxPooling
            elseif isa(layer, ConvolutionalLayers.MaxPooling)
                info["pool_size"] = layer.pool_size
                info["stride"] = layer.stride
                
            # Embedding
            elseif isa(layer, EmbeddingLayer.Embedding)
                info["num_embeddings"] = size(layer.weights.data, 1)
                info["embedding_dim"] = size(layer.weights.data, 2)
                
            # Flatten
            elseif isa(layer, Layers.Flatten)
                info["layer_type"] = "Flatten"
            elseif isa(layer, Layers.LayerNorm)
                info["num_features"] = layer.normalized_shape  # Tupla completa
                info["eps"] = layer.eps
                # Guardar parámetros en la estructura
                info["gamma"] = Array(layer.gamma.data)
                info["beta"] = Array(layer.beta.data)
            end
            
            push!(layer_info, info)
        end
    end
    
    return layer_info
end

function get_architecture_string(model)
    if isa(model, NeuralNetwork.Sequential)
        return "Sequential"
    else
        return string(typeof(model))
    end
end

function create_model_data(model)
    params = NeuralNetwork.collect_parameters(model)
    weights = []
    
    for p in params
        if p.data isa CUDA.CuArray
            push!(weights, Array(p.data))
        else
            push!(weights, copy(p.data))
        end
    end
    
    layer_info = extract_architecture(model)
    
    metadata = Dict{String,Any}(
        "version" => "DeepDynamics-1.0",
        "date" => string(Dates.now()),
        "device" => get_model_device(model)
    )
    
    return ModelData(get_architecture_string(model), weights, layer_info, metadata)
end

function get_model_device(model)
    params = NeuralNetwork.collect_parameters(model)
    if !isempty(params) && params[1].data isa CUDA.CuArray
        return "cuda"
    else
        return "cpu"
    end
end

function reconstruct_model(layer_info)
    layers = []
    
    for info in layer_info
        layer_type = info["type"]
        
        if contains(layer_type, "Dense")
            layer = NeuralNetwork.Dense(
                info["in_features"],
                info["out_features"]
            )
        elseif contains(layer_type, "LayerNorm")
            shape = Tuple(info["num_features"])  # Convertir de vuelta a tupla
            layer = Layers.LayerNorm(
                shape;
                eps=get(info, "eps", 1f-5)
            )
            # Restaurar parámetros si existen
            if haskey(info, "gamma")
                layer.gamma.data .= info["gamma"]
                layer.beta.data .= info["beta"]
            end


        elseif contains(layer_type, "RNNCell")
            layer = Layers.RNNCell(
                info["input_size"],
                info["hidden_size"];
                bias = info["has_bias"]
            )

        elseif contains(layer_type, "RNN")
            cell_info = info["cell_info"]
            cell = Layers.RNNCell(
                cell_info["input_size"],
                cell_info["hidden_size"];
                bias = cell_info["has_bias"]
            )
            layer = Layers.RNN(cell.input_size, cell.hidden_size;
                            batch_first = info["batch_first"],
                            return_sequences = info["return_sequences"])
            layer.cell = cell

        elseif contains(layer_type, "Conv2D")
            layer = ConvolutionalLayers.Conv2D(
                info["in_channels"],
                info["out_channels"], 
                info["kernel_size"];
                stride=info["stride"],
                padding=info["padding"]
            )
        elseif contains(layer_type, "BatchNorm")
            layer = Layers.BatchNorm(
                info["num_features"];
                momentum=info["momentum"],
                epsilon=info["epsilon"]
            )
        
            if haskey(info, "running_mean")
                layer.running_mean = copy(info["running_mean"])
                layer.running_var = copy(info["running_var"])
                layer.num_batches_tracked = info["num_batches_tracked"]
            end
        elseif contains(layer_type, "DropoutLayer")
            layer = Layers.DropoutLayer(info["rate"])
        elseif contains(layer_type, "Activation")
            act_name = info["activation"]
            if contains(act_name, "relu")
                layer = NeuralNetwork.Activation(NeuralNetwork.relu)
            elseif contains(act_name, "sigmoid")
                layer = NeuralNetwork.Activation(NeuralNetwork.sigmoid)
            elseif contains(act_name, "tanh")
                layer = NeuralNetwork.Activation(NeuralNetwork.tanh_activation)
            elseif contains(act_name, "softmax")
                layer = NeuralNetwork.Activation(NeuralNetwork.softmax)
            elseif contains(act_name, "leaky_relu")
                layer = NeuralNetwork.Activation(NeuralNetwork.leaky_relu)
            elseif contains(act_name, "swish")
                layer = NeuralNetwork.Activation(NeuralNetwork.swish)
            elseif contains(act_name, "mish")
                layer = NeuralNetwork.Activation(NeuralNetwork.mish)
            else
                layer = NeuralNetwork.Activation(NeuralNetwork.relu)
            end
        elseif contains(layer_type, "Flatten")
            layer = Layers.Flatten()
        elseif contains(layer_type, "MaxPooling")
            layer = ConvolutionalLayers.MaxPooling(
                info["pool_size"];
                stride=info["stride"]
            )
        elseif contains(layer_type, "Embedding")
            layer = EmbeddingLayer.Embedding(
                info["num_embeddings"],
                info["embedding_dim"]
            )
        else
            @warn "Tipo de capa desconocido: $layer_type"
            continue
        end
        
        push!(layers, layer)
    end
    
    return NeuralNetwork.Sequential(layers)
end

function reconstruct_from_model_data(model_data::ModelData)
    model = reconstruct_model(model_data.layer_info)
    
    params = NeuralNetwork.collect_parameters(model)
    for (param, weight) in zip(params, model_data.weights)
        param.data = copy(weight)
    end
    
    return model
end

function extract_optimizer_state(optimizer, model)
    state = Dict{String,Any}("type" => string(typeof(optimizer)))
    
    params = NeuralNetwork.collect_parameters(model)
    param_to_idx = Dict(p => i for (i, p) in enumerate(params))
    
    if isa(optimizer, Optimizers.SGD)
        state["learning_rate"] = optimizer.learning_rate
        
    elseif isa(optimizer, Optimizers.Adam)
        state["learning_rate"] = optimizer.learning_rate
        state["beta1"] = optimizer.beta1
        state["beta2"] = optimizer.beta2
        state["epsilon"] = optimizer.epsilon
        state["weight_decay"] = optimizer.weight_decay
        state["t"] = optimizer.t
        
        state["m_indices"] = Int[]
        state["m_values"] = []
        state["v_indices"] = Int[]
        state["v_values"] = []
        
        for (param, m_tensor) in optimizer.m
            if haskey(param_to_idx, param)
                idx = param_to_idx[param]
                push!(state["m_indices"], idx)
                push!(state["m_values"], m_tensor.data isa CUDA.CuArray ? Array(m_tensor.data) : copy(m_tensor.data))
            end
        end
        
        for (param, v_tensor) in optimizer.v
            if haskey(param_to_idx, param)
                idx = param_to_idx[param]
                push!(state["v_indices"], idx)
                push!(state["v_values"], v_tensor.data isa CUDA.CuArray ? Array(v_tensor.data) : copy(v_tensor.data))
            end
        end
        
    elseif isa(optimizer, Optimizers.RMSProp)
        state["learning_rate"] = optimizer.learning_rate
        state["decay_rate"] = optimizer.decay_rate
        state["epsilon"] = optimizer.epsilon
        
        state["cache_indices"] = Int[]
        state["cache_values"] = []
        
        for (param, cache_tensor) in optimizer.cache
            if haskey(param_to_idx, param)
                idx = param_to_idx[param]
                push!(state["cache_indices"], idx)
                push!(state["cache_values"], cache_tensor.data isa CUDA.CuArray ? Array(cache_tensor.data) : copy(cache_tensor.data))
            end
        end
        
    elseif isa(optimizer, Optimizers.Adagrad)
        state["learning_rate"] = optimizer.learning_rate
        state["epsilon"] = optimizer.epsilon
        
        state["cache_indices"] = Int[]
        state["cache_values"] = []
        
        for (param, cache_tensor) in optimizer.cache
            if haskey(param_to_idx, param)
                idx = param_to_idx[param]
                push!(state["cache_indices"], idx)
                push!(state["cache_values"], cache_tensor.data isa CUDA.CuArray ? Array(cache_tensor.data) : copy(cache_tensor.data))
            end
        end
        
    elseif isa(optimizer, Optimizers.Nadam)
        state["learning_rate"] = optimizer.learning_rate
        state["beta1"] = optimizer.beta1
        state["beta2"] = optimizer.beta2
        state["epsilon"] = optimizer.epsilon
        state["weight_decay"] = optimizer.weight_decay
        state["t"] = optimizer.t
        
        state["m_indices"] = Int[]
        state["m_values"] = []
        state["v_indices"] = Int[]
        state["v_values"] = []
        
        for (param, m_tensor) in optimizer.m
            if haskey(param_to_idx, param)
                idx = param_to_idx[param]
                push!(state["m_indices"], idx)
                push!(state["m_values"], m_tensor.data isa CUDA.CuArray ? Array(m_tensor.data) : copy(m_tensor.data))
            end
        end
        
        for (param, v_tensor) in optimizer.v
            if haskey(param_to_idx, param)
                idx = param_to_idx[param]
                push!(state["v_indices"], idx)
                push!(state["v_values"], v_tensor.data isa CUDA.CuArray ? Array(v_tensor.data) : copy(v_tensor.data))
            end
        end
    end
    
    return state
end

function reconstruct_optimizer(state::Dict{String,Any}, model)
    opt_type = state["type"]
    params = NeuralNetwork.collect_parameters(model)
    
    if contains(opt_type, "SGD")
        return Optimizers.SGD(learning_rate=state["learning_rate"])
        
    elseif contains(opt_type, "Adam")
        opt = Optimizers.Adam(
            learning_rate=state["learning_rate"],
            beta1=state["beta1"],
            beta2=state["beta2"],
            epsilon=state["epsilon"],
            weight_decay=state["weight_decay"]
        )
        opt.t = state["t"]
        
        for (idx, m_val) in zip(state["m_indices"], state["m_values"])
            if idx <= length(params)
                param = params[idx]
                opt.m[param] = TensorEngine.Tensor(copy(m_val))
            end
        end
        
        for (idx, v_val) in zip(state["v_indices"], state["v_values"])
            if idx <= length(params)
                param = params[idx]
                opt.v[param] = TensorEngine.Tensor(copy(v_val))
            end
        end
        
        return opt
        
    elseif contains(opt_type, "RMSProp")
        opt = Optimizers.RMSProp(
            learning_rate=state["learning_rate"],
            decay_rate=state["decay_rate"],
            epsilon=state["epsilon"]
        )
        
        for (idx, cache_val) in zip(state["cache_indices"], state["cache_values"])
            if idx <= length(params)
                param = params[idx]
                opt.cache[param] = TensorEngine.Tensor(copy(cache_val))
            end
        end
        
        return opt
        
    elseif contains(opt_type, "Adagrad")
        opt = Optimizers.Adagrad(
            learning_rate=state["learning_rate"],
            epsilon=state["epsilon"]
        )
        
        for (idx, cache_val) in zip(state["cache_indices"], state["cache_values"])
            if idx <= length(params)
                param = params[idx]
                opt.cache[param] = TensorEngine.Tensor(copy(cache_val))
            end
        end
        
        return opt
        
    elseif contains(opt_type, "Nadam")
        opt = Optimizers.Nadam(
            learning_rate=state["learning_rate"],
            beta1=state["beta1"],
            beta2=state["beta2"],
            epsilon=state["epsilon"],
            weight_decay=state["weight_decay"]
        )
        opt.t = state["t"]
        
        for (idx, m_val) in zip(state["m_indices"], state["m_values"])
            if idx <= length(params)
                param = params[idx]
                opt.m[param] = TensorEngine.Tensor(copy(m_val))
            end
        end
        
        for (idx, v_val) in zip(state["v_indices"], state["v_values"])
            if idx <= length(params)
                param = params[idx]
                opt.v[param] = TensorEngine.Tensor(copy(v_val))
            end
        end
        
        return opt
        
    else
        @warn "Tipo de optimizador desconocido: $opt_type, usando SGD por defecto"
        return Optimizers.SGD(learning_rate=0.01)
    end
end

# ==================================================================
# FUNCIONES AUXILIARES NUEVAS
# ==================================================================

function create_model_bundle(model, version::VersionNumber, metadata::Dict, 
                           include_source::Bool, compression::Symbol)
    # Usar las funciones existentes
    model_data = create_model_data(model)
    
    # Extraer arquitectura más detallada
    architecture = Dict{String,Any}(
        "type" => model_data.architecture,
        "layers" => model_data.layer_info
    )
    
    # Convertir weights a Dict
    weights_dict = Dict{String,Any}()
    params = NeuralNetwork.collect_parameters(model)
    for (i, weight) in enumerate(model_data.weights)
        weights_dict["param_$i"] = weight
    end
    
    # Calcular checksum
    checksum = calculate_weights_checksum(weights_dict)
    
    # Metadata completa
    full_metadata = merge(model_data.metadata, metadata)
    full_metadata["compression"] = string(compression)
    full_metadata["bundle_version"] = "1.0"
    
    if include_source
        full_metadata["source_code"] = "# Código fuente no implementado aún"
    end
    
    ModelBundle(
        version,
        architecture,
        weights_dict,
        full_metadata,
        nothing,
        nothing,
        checksum,
        compression
    )
end

function save_model_jld2(filepath::String, bundle::ModelBundle, compression::Symbol)
    if compression == :none
        # Sin compresión - guardar normal
        jldopen(filepath, "w"; compress=false) do file
            file["bundle_version"] = "1.0"
            file["model_version"] = string(bundle.version)
            file["architecture"] = bundle.architecture
            file["weights"] = bundle.weights
            file["metadata"] = bundle.metadata
            file["checksum"] = bundle.checksum
            file["compression"] = "none"
        end
    else
        # CON compresión - guardar de forma diferente
        # Primero serializar los weights a bytes
        io = IOBuffer()
        serialize(io, bundle.weights)
        weight_bytes = take!(io)
        
        # Comprimir los bytes
        compressed_bytes = if compression == :zstd
            transcode(ZstdCompressor, weight_bytes)
        elseif compression == :lz4
            transcode(LZ4FrameCompressor, weight_bytes)
        end
        
        # Guardar solo los bytes comprimidos, no objetos Dict
        jldopen(filepath, "w"; compress=false) do file
            file["bundle_version"] = "1.0"
            file["model_version"] = string(bundle.version)
            file["architecture"] = bundle.architecture
            file["compressed_weights"] = compressed_bytes  # Solo bytes
            file["metadata"] = bundle.metadata
            file["checksum"] = bundle.checksum
            file["compression"] = string(compression)
        end
    end
    
    println("✅ Modelo guardado en formato JLD2: $filepath")
end

function load_model_jld2(filepath::String, lazy_load::Bool)
    bundle = nothing
    
    try
        jldopen(filepath, "r") do file
            # Verificar estructura
            if !haskey(file, "bundle_version")
                error("Archivo JLD2 no es un modelo válido de DeepDynamics")
            end
            
            version = VersionNumber(file["model_version"])
            architecture = file["architecture"]
            metadata = file["metadata"]
            checksum = file["checksum"]
            compression = Symbol(file["compression"])
            
            # Cargar weights según formato
            weights = if compression == :none
                file["weights"]
            else
                # Descomprimir bytes
                compressed_bytes = file["compressed_weights"]
                
                decompressed = if compression == :zstd
                    transcode(ZstdDecompressor, compressed_bytes)
                elseif compression == :lz4
                    transcode(LZ4FrameDecompressor, compressed_bytes)
                end
                
                # Deserializar
                io = IOBuffer(decompressed)
                deserialize(io)
            end
            
            bundle = ModelBundle(
                version,
                architecture,
                weights,
                metadata,
                nothing,
                nothing,
                checksum,
                compression
            )
        end
    catch e
        if isa(e, EOFError)
            error("Error cargando modelo: archivo corrupto o formato inválido")
        else
            rethrow(e)
        end
    end
    
    return bundle
end

function save_model_bson(filepath::String, bundle::ModelBundle, compression::Symbol)
    # Por ahora, usar BSON básico
    data = Dict(
        "bundle" => bundle
    )
    BSON.@save filepath data
    println("✅ Modelo guardado en formato BSON: $filepath")
end

function load_model_bson(filepath::String, lazy_load::Bool)
    BSON.@load filepath data
    return data["bundle"]
end

function detect_format(filepath::String)
    if endswith(filepath, ".jld2")
        return :jld2
    elseif endswith(filepath, ".bson")
        return :bson
    else
        # Intentar detectar por contenido
        try
            jldopen(filepath, "r") do file
                haskey(file, "bundle_version") && return :jld2
            end
        catch
        end
        
        return :serialization  # Default al formato original
    end
end

function calculate_weights_checksum(weights::Dict)
    # Concatenar todos los pesos
    all_data = Float32[]
    for key in sort(collect(keys(weights)))
        append!(all_data, vec(weights[key]))
    end
    
    # Calcular SHA256
    bytes = reinterpret(UInt8, all_data)
    return bytes2hex(sha256(bytes))
end

function verify_checksum(bundle::ModelBundle)
    calculated = calculate_weights_checksum(bundle.weights)
    if calculated != bundle.checksum
        error("Checksum no coincide. El modelo puede estar corrupto.")
    end
end

function compress_weights(weights::Dict, method::Symbol)
    compressed = Dict{String,Any}()
    
    for (key, value) in weights
        bytes = collect(reinterpret(UInt8, vec(value)))
        
        if method == :zstd
            compressed[key] = transcode(ZstdCompressor, bytes)
        elseif method == :lz4
            compressed[key] = transcode(LZ4FrameCompressor, bytes)  # <- Nombre correcto
        end
        
        compressed["$(key)_shape"] = size(value)
        compressed["$(key)_type"] = eltype(value)
    end
    
    return compressed
end

function decompress_weights(compressed::Dict, method::Symbol)
    weights = Dict{String,Any}()
    
    weight_keys = filter(k -> !endswith(k, "_shape") && !endswith(k, "_type"), 
                        keys(compressed))
    
    for key in weight_keys
        bytes = compressed[key]
        shape = compressed["$(key)_shape"]
        dtype = compressed["$(key)_type"]
        
        decompressed_bytes = if method == :zstd
            transcode(ZstdDecompressor, bytes)
        elseif method == :lz4
            transcode(LZ4FrameDecompressor, bytes)  # <- Nombre correcto
        else
            bytes
        end
        
        flat_array = collect(reinterpret(dtype, decompressed_bytes))
        weights[key] = reshape(flat_array, shape)
    end
    
    return weights
end

function migrate_bundle(bundle::ModelBundle)
    # Por ahora no hay migraciones necesarias
    # En el futuro aquí se implementarán migraciones entre versiones
    return bundle
end

function reconstruct_from_bundle(bundle::ModelBundle)
    # Reconstruir usando las funciones existentes
    layer_info = bundle.architecture["layers"]
    model = reconstruct_model(layer_info)
    
    # Restaurar pesos
    params = NeuralNetwork.collect_parameters(model)
    for (i, param) in enumerate(params)
        key = "param_$i"
        if haskey(bundle.weights, key)
            param.data = copy(bundle.weights[key])
        end
    end
    
    return model
end

# ==================================================================
# MODEL REGISTRY
# ==================================================================

function register_model(registry::ModelRegistry, model, name::String, tags::Vector{String}=String[];
                       version::String="latest", metadata::Dict=Dict())
    timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")
    filename = "$(name)_$(version)_$(timestamp).jld2"
    filepath = joinpath(registry.base_path, filename)
    
    # Guardar modelo
    bundle = save_model(model, filepath; 
                       format=:jld2, 
                       compression=true,
                       metadata=metadata)
    
    # Crear info
    model_info = ModelInfo(
        name,
        filepath,
        version,
        tags,
        metadata,
        now(),
        filesize(filepath),
        bundle.checksum
    )
    
    # Actualizar registry
    if !haskey(registry.models, name)
        registry.models[name] = ModelInfo[]
    end
    push!(registry.models[name], model_info)
    
    save_registry_index(registry)
    
    println("✅ Modelo registrado: $name v$version")
    return model_info
end

function get_model(registry::ModelRegistry, name::String; version::String="latest")
    if !haskey(registry.models, name)
        error("Modelo '$name' no encontrado")
    end
    
    versions = registry.models[name]
    
    model_info = if version == "latest"
        sort(versions, by=x->x.created_at, rev=true)[1]
    else
        found = filter(x -> x.version == version, versions)
        isempty(found) ? error("Versión no encontrada") : found[1]
    end
    
    return load_model(model_info.path), model_info
end

function list_models(registry::ModelRegistry; filter_tags::Vector{String}=String[])
    models = []
    
    for (name, versions) in registry.models
        for info in versions
            if isempty(filter_tags) || any(tag in info.tags for tag in filter_tags)
                push!(models, info)
            end
        end
    end
    
    sort!(models, by=x->x.created_at, rev=true)
    return models
end

function load_registry_index(index_file::String)
    if isfile(index_file)
        data = JSON.parsefile(index_file)
        models = Dict{String,Vector{ModelInfo}}()
        
        for (name, versions) in data
            models[name] = [ModelInfo(
                info["name"],
                info["path"],
                info["version"],
                info["tags"],
                info["metadata"],
                DateTime(info["created_at"]),
                info["file_size"],
                info["checksum"]
            ) for info in versions]
        end
        
        return models
    else
        return Dict{String,Vector{ModelInfo}}()
    end
end

function save_registry_index(registry::ModelRegistry)
    data = Dict{String,Any}()
    
    for (name, versions) in registry.models
        data[name] = [Dict(
            "name" => info.name,
            "path" => info.path,
            "version" => info.version,
            "tags" => info.tags,
            "metadata" => info.metadata,
            "created_at" => string(info.created_at),
            "file_size" => info.file_size,
            "checksum" => info.checksum
        ) for info in versions]
    end
    
    open(registry.index_file, "w") do f
        JSON.print(f, data, 2)
    end
end

end # module