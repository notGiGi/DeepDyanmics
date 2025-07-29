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

export save_model, load_model, save_checkpoint, load_checkpoint

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

function load_model(filepath::String; device="auto")
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
    
    if haskey(model_data.metadata, "version")
        version = model_data.metadata["version"]
        if version != "DeepDynamics-1.0"
            @warn "Versión diferente: $version, pueden haber incompatibilidades"
        end
    end
    
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
    
    if device == "auto"
        device = get(model_data.metadata, "device", "cpu")
    end
    
    if device == "cuda" && CUDA.functional()
        model = NeuralNetwork.model_to_gpu(model)
    elseif device == "cpu"
        model = NeuralNetwork.model_to_cpu(model)
    end
    
    println("✅ Modelo cargado desde: $filepath")
    println("   Dispositivo: $device")
    println("   Parámetros: $(model_data.metadata["num_parameters"])")
    
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

end # module