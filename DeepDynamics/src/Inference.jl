# src/Inference.jl
module Inference

using CUDA
using Statistics
using ..TensorEngine
using ..NeuralNetwork
using ..Layers
using ..ConvolutionalLayers

export predict, predict_proba, predict_generator, PredictionPipeline,
       quantize_model, warmup_model, TensorPool, get_tensor_from_pool!,
       return_to_pool!

# ============= FUNCIONES AUXILIARES =============

"""
Establece modo training/eval para todas las capas
"""

function set_inference_mode!(model, training::Bool)
    if isa(model, NeuralNetwork.Sequential)
        for layer in model.layers
            set_inference_mode!(layer, training)
        end
    elseif isa(model, Layers.BatchNorm) || isa(model, Layers.DropoutLayer)
        model.training = training  # ✅ Cambiar 'layer' por 'model'
    end
    return model
end

"""
Verifica disponibilidad de GPU
"""
function check_gpu_available()
    return CUDA.functional() && CUDA.has_cuda_gpu()
end

# ============= PREDICT PRINCIPAL =============

"""
    predict(model, X; batch_size=32, device=:auto, verbose=false)

Realiza predicción en lotes de datos.
- X: Datos de entrada (Array o Vector de Tensores)
- batch_size: Tamaño del batch para procesamiento
- device: :auto (detecta automáticamente), :cpu o :cuda
- verbose: Mostrar progreso
"""
function predict(model, X; 
                batch_size::Int=32, 
                device::Symbol=:auto,
                verbose::Bool=false)
    
    # Validar entrada no vacía
    if (X isa AbstractArray && length(X) == 0) || (X isa Vector && isempty(X))
        throw(ArgumentError("Input data cannot be empty"))
    end
    
    # Configurar dispositivo
    use_gpu = if device == :auto
        check_gpu_available()
    elseif device == :cuda
        !CUDA.functional() && error("CUDA requested but not available")
        true
    else
        false
    end
    
    verbose && println("Prediction using: ", use_gpu ? "GPU" : "CPU")
    
    # Establecer modo evaluación
    set_inference_mode!(model, false)
    
    # Preparar datos en batches
    input_tensors = prepare_batches(X, batch_size)
    
    # Mover modelo a dispositivo apropiado
    model_device = use_gpu ? NeuralNetwork.model_to_gpu(model) : NeuralNetwork.model_to_cpu(model)
    
    # Realizar predicciones
    predictions = []
    for (i, batch) in enumerate(input_tensors)
        verbose && i % 10 == 0 && println("  Processing batch $i/$(length(input_tensors))")
        
        # Mover batch a dispositivo
        batch_device = use_gpu ? 
            TensorEngine.to_gpu(batch) : 
            TensorEngine.to_cpu(batch)
        
        # Forward pass sin gradientes
        pred = NeuralNetwork.forward(model_device, batch_device; verbose=false)
        
        # Mover resultado a CPU para output
        pred_cpu = TensorEngine.to_cpu(pred)
        push!(predictions, pred_cpu.data)
    end
    
    # Concatenar resultados
    return length(predictions) == 1 ? predictions[1] : cat(predictions..., dims=ndims(predictions[1]))
end

"""
Prepara los datos en batches para procesamiento
"""

function prepare_batches(X, batch_size::Int)
    if X isa AbstractArray && !(X isa Vector{<:TensorEngine.Tensor})
        # Array multidimensional normal
        n_samples = size(X)[end]
        return [TensorEngine.Tensor(selectdim(X, ndims(X), i:min(i+batch_size-1, n_samples))) 
                for i in 1:batch_size:n_samples]
    elseif X isa Vector{<:TensorEngine.Tensor}
        # Vector de tensores - NO usar comprehension con SubArray
        batches = []
        i = 1
        while i <= length(X)
            batch_end = min(i + batch_size - 1, length(X))
            
            if i == batch_end  # Solo un tensor
                push!(batches, X[i])
            else
                # Extraer datos y concatenar
                tensor_group = X[i:batch_end]
                batch_data = hcat([t.data for t in tensor_group]...)
                push!(batches, TensorEngine.Tensor(batch_data))
            end
            i = batch_end + 1
        end
        return batches
    else
        error("Unsupported input format: $(typeof(X))")
    end
end

# ============= PREDICT PROBA =============

"""
    predict_proba(model, X; temperature=1.0, kwargs...)

Retorna probabilidades para clasificación.
Temperature scaling permite calibración de confianza.
"""
function predict_proba(model, X; 
                      temperature::Float32=1.0f0,
                      batch_size::Int=32,
                      device::Symbol=:auto,
                      verbose::Bool=false)
    
    # Obtener logits usando predict
    logits = predict(model, X; batch_size=batch_size, device=device, verbose=verbose)
    
    # Aplicar temperature scaling
    if temperature != 1.0f0
        logits = logits ./ temperature
    end
    
    # Convertir a probabilidades - usar función sigmoid estándar de Julia
    if ndims(logits) == 1 || size(logits, 1) == 1
        # Clasificación binaria - usar función sigmoid matemática
        return @. 1.0f0 / (1.0f0 + exp(-logits))
    else
        # Multiclase - aplicar softmax
        exp_logits = exp.(logits .- maximum(logits, dims=1))
        return exp_logits ./ sum(exp_logits, dims=1)
    end
end

# ============= PREDICT GENERATOR =============

"""
    predict_generator(model, data_generator; steps, workers, use_multiprocessing)

Predicción usando generador de datos para datasets grandes.
Soporta procesamiento paralelo con múltiples workers.
"""

function predict_generator(model, data_generator;
                         steps::Union{Nothing,Int}=nothing,
                         workers::Int=1,
                         use_multiprocessing::Bool=false)
    
    set_inference_mode!(model, false)
    
    # Determinar dispositivo
    use_gpu = check_gpu_available()
    model_device = use_gpu ? NeuralNetwork.model_to_gpu(model) : NeuralNetwork.model_to_cpu(model)
    
    
    if use_multiprocessing && workers > 1 && Threads.nthreads() > 1
        # Procesamiento paralelo
        predictions = process_parallel(model_device, data_generator, steps, workers, use_gpu)
    else
        # Procesamiento secuencial
        predictions = process_sequential(model_device, data_generator, steps, use_gpu)
    end
    
    # Manejar caso de predicciones vacías
    if isempty(predictions)
        return Float32[]
    end
    
    return cat(predictions..., dims=ndims(predictions[1]))
end


function process_sequential(model, data_generator, steps, use_gpu)
    predictions = []
    step_count = 0
    
    for batch in data_generator
        batch_tensor = TensorEngine.Tensor(batch)
        batch_device = use_gpu ? TensorEngine.to_gpu(batch_tensor) : batch_tensor
        
        pred = NeuralNetwork.forward(model, batch_device; verbose=false)
        push!(predictions, TensorEngine.to_cpu(pred).data)
        
        step_count += 1
        if steps !== nothing && step_count >= steps
            break
        end
    end
    
    # Si no hay predicciones, retornar array vacío con dimensiones correctas
    if isempty(predictions)
        return Float32[]
    end
    
    return predictions
end

function process_parallel(model, data_generator, steps, workers, use_gpu)
    batch_channel = Channel(workers * 2)
    results = Channel(workers * 2)
    step_count = Threads.Atomic{Int}(0)
    
    # Productor
    @async begin
        for batch in data_generator
            if steps !== nothing && step_count[] >= steps
                break
            end
            put!(batch_channel, batch)
            Threads.atomic_add!(step_count, 1)
        end
        close(batch_channel)
    end
    
    # Consumidores
    @sync for _ in 1:workers
        Threads.@spawn begin
            for batch in batch_channel
                batch_device = use_gpu ? 
                    TensorEngine.to_gpu(TensorEngine.Tensor(batch)) : 
                    TensorEngine.Tensor(batch)
                
                pred = NeuralNetwork.forward(model, batch_device; verbose=false)
                put!(results, TensorEngine.to_cpu(pred).data)
            end
        end
    end
    close(results)
    
    # Recolectar resultados
    return collect(results)
end

# ============= PIPELINE =============

"""
    PredictionPipeline

Pipeline completo con preprocesamiento y postprocesamiento.
"""
struct PredictionPipeline
    model::Any
    preprocessor::Function
    postprocessor::Function
    device::Symbol
    batch_size::Int
    
    function PredictionPipeline(model, preprocessor=identity, postprocessor=identity;
                               device::Symbol=:auto, batch_size::Int=32)
        new(model, preprocessor, postprocessor, device, batch_size)
    end
end

# Hacer callable el pipeline
function (pipeline::PredictionPipeline)(X)
    # Preprocesar
    X_processed = pipeline.preprocessor(X)
    
    # Predecir
    predictions = predict(pipeline.model, X_processed; 
                        batch_size=pipeline.batch_size,
                        device=pipeline.device,
                        verbose=false)
    
    # Postprocesar
    return pipeline.postprocessor(predictions)
end

# ============= OPTIMIZACIONES =============

"""
    quantize_model(model; bits=8)

Cuantización del modelo para inferencia rápida.
Reduce precisión de pesos a int8/int16 para menor uso de memoria.
"""
function quantize_model(model; bits::Int=8)
    quantized = deepcopy(model)
    scale = Float32(2^(bits-1) - 1)
    
    for layer in quantized.layers
        # Cuantizar pesos
        if hasproperty(layer, :weights)
            W = layer.weights.data
            W_max = maximum(abs.(W), dims=2:ndims(W))
            W_scale = W_max ./ scale .+ 1f-8
            W_quant = round.(Int8, W ./ W_scale)
            
            layer.weights = TensorEngine.Tensor(Float32.(W_quant) .* W_scale; 
                                               requires_grad=false)
        end
        
        # Convertir bias a FP16
        if hasproperty(layer, :biases)
            layer.biases = TensorEngine.Tensor(Float16.(layer.biases.data); 
                                              requires_grad=false)
        elseif hasproperty(layer, :bias)
            layer.bias = TensorEngine.Tensor(Float16.(layer.bias.data); 
                                            requires_grad=false)
        end
    end
    
    return quantized
end

"""
    warmup_model(model, sample_input; n_runs=3, verbose=true)

Calienta el caché del modelo antes de inferencia.
"""

# REEMPLAZAR función warmup_model:
function warmup_model(model, sample_input; n_runs::Int=3, verbose::Bool=true)
    verbose && println("Warming up model...")
    
    # Obtener muestra
    sample = if isa(sample_input, Function) || applicable(iterate, sample_input)
        first(sample_input)
    else
        sample_input
    end
    
    # Convertir a tensor con dimensiones correctas
    sample_tensor = if isa(sample, TensorEngine.Tensor)
        # Ya es tensor, verificar dimensiones
        if ndims(sample.data) == 0
            # Escalar - expandir a matriz columna mínima
            TensorEngine.Tensor(reshape([sample.data], 1, 1))
        elseif ndims(sample.data) == 1
            # Vector - expandir a matriz columna
            TensorEngine.Tensor(reshape(sample.data, :, 1))
        else
            sample
        end
    elseif isa(sample, AbstractArray)
        if ndims(sample) == 0
            # Escalar
            TensorEngine.Tensor(reshape([sample], 1, 1))
        elseif ndims(sample) == 1
            # Vector - convertir a matriz columna
            TensorEngine.Tensor(reshape(sample, :, 1))
        else
            # Matriz o tensor multidimensional
            TensorEngine.Tensor(sample)
        end
    else
        # Intentar convertir a array
        data = collect(sample)
        if ndims(data) == 1
            TensorEngine.Tensor(reshape(data, :, 1))
        else
            TensorEngine.Tensor(data)
        end
    end
    
    # Ejecutar forward passes
    for i in 1:n_runs
        try
            _ = NeuralNetwork.forward(model, sample_tensor; verbose=false)
        catch e
            if verbose
                println("Warmup failed with sample size: ", size(sample_tensor.data))
                rethrow(e)
            end
        end
    end
    
    verbose && println("Model warmed up ✓")
end

# ============= MEMORY POOL =============

"""
    TensorPool

Pool de tensores para reutilización de memoria.
"""
mutable struct TensorPool
    pools::Dict{Tuple, Vector{TensorEngine.Tensor}}
    max_tensors_per_size::Int
end

TensorPool(; max_tensors::Int=10) = TensorPool(Dict(), max_tensors)

"""
Obtiene un tensor del pool o crea uno nuevo
"""
function get_tensor_from_pool!(pool::TensorPool, shape::Tuple, device::Symbol)
    key = shape
    
    if haskey(pool.pools, key) && !isempty(pool.pools[key])
        tensor = pop!(pool.pools[key])
        # Limpiar datos
        tensor.data .= 0
        return tensor
    else
        # Crear nuevo
        data = zeros(Float32, shape...)
        tensor = TensorEngine.Tensor(data; requires_grad=false)
        return device == :cuda ? TensorEngine.to_gpu(tensor) : tensor
    end
end

"""
Devuelve un tensor al pool para reutilización
"""
function return_to_pool!(pool::TensorPool, tensor::TensorEngine.Tensor)
    key = size(tensor.data)
    
    if !haskey(pool.pools, key)
        pool.pools[key] = TensorEngine.Tensor[]
    end
    
    if length(pool.pools[key]) < pool.max_tensors_per_size
        push!(pool.pools[key], tensor)
    end
end

end # module Inference