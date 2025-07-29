module DeepDynamics
using CUDA
# ----------------------------------------------------------
# 🚀 Inicialización y detección de GPU
# ----------------------------------------------------------
const GPU_AVAILABLE = Ref{Bool}(false)
const GPU_DEVICE = Ref{String}("cpu")

function __init__()
    try
        if CUDA.functional()
            GPU_AVAILABLE[] = true
            GPU_DEVICE[] = "cuda"
            device_info = CUDA.device()
            gpu_name = CUDA.name(device_info)
            gpu_memory = CUDA.totalmem(device_info) / 1e9
            cuda_version = try string(CUDA.runtime_version()) catch _ "desconocida" end

            @info """
            🚀 DeepDynamics.jl inicializado con soporte GPU
            ✓ GPU detectada: $gpu_name
            ✓ Memoria disponible: $(round(gpu_memory, digits=2)) GB
            ✓ CUDA versión: $cuda_version
            """
        else
            @info """
            ⚡ DeepDynamics.jl inicializado en modo CPU
            ℹ️  GPU no disponible o CUDA no funcional
            """
        end
    catch e
        @info """
        ⚡ DeepDynamics.jl inicializado en modo CPU
        ℹ️  CUDA no disponible: $e
        """
    end
end


# ----------------------------------------------------------
# API para acceso a estado del dispositivo
# ----------------------------------------------------------
export gpu_available, get_device, set_default_device!, auto_device, validate_gpu_environment

gpu_available() = GPU_AVAILABLE[]
get_device() = GPU_DEVICE[]

function set_default_device!(device::String)
    if device == "cuda" && !GPU_AVAILABLE[]
        @warn "GPU no disponible, manteniendo CPU como dispositivo"
        return
    end
    GPU_DEVICE[] = device
    @info "Dispositivo por defecto establecido a: $device"
end

"""
    auto_device(x)

Envía un tensor o array a GPU si está disponible, o lo mantiene en CPU.
"""
function auto_device(x)
    return gpu_available() ? to_gpu(x) : x
end

"""
    validate_gpu_environment()

Verifica si CUDA está operativo y lanza advertencia si se usa entrenamiento GPU sin soporte real.
"""
function validate_gpu_environment()
    if get_device() == "cuda" && !gpu_available()
        @warn "Se configuró CUDA como dispositivo, pero no se detectó GPU funcional. Revirtiendo a CPU."
        set_default_device!("cpu")
    end
end

# ----------------------------------------------------------
# Inclusión de submódulos
# ----------------------------------------------------------
include("GPUMemoryManager.jl")
include("DataAugmentation.jl")
include("AbstractLayer.jl")
include("TensorEngine.jl")
include("ConvKernelLayers.jl")
include("Visualizations.jl")
include("ReshapeModule.jl")

include("ConvolutionalLayers.jl")
include("Layers.jl")
include("EmbeddingLayer.jl")
include("NeuralNetwork.jl")
include("Losses.jl")
include("Utils.jl")
include("Optimizers.jl")
include("ModelSaver.jl")
include("Callbacks.jl")
include("Metrics.jl")
include("Reports.jl")
include("DataLoaders.jl")
include("Training.jl")
include("TextUtils.jl")
include("ImageProcessing.jl")
include("UNetSegmentation.jl")
include("PretrainedModels.jl")
include("LRSchedulers.jl")
include("Models.jl")

# ----------------------------------------------------------
# Imports y Reexports principales
# ----------------------------------------------------------
using .TensorEngine: Tensor, backward, mse_loss, initialize_grad!, initialize_weights,
                     l2_regularization, compute_loss_with_regularization, clip_gradients!,
                     to_gpu, to_cpu, zero_grad!, add, device_of, same_device, gpu_memory_info,
                     ensure_gpu_memory!, log, mean

using .Visualizations
using .ReshapeModule: Reshape
using .Losses: binary_crossentropy, categorical_crossentropy,binary_crossentropy_with_logits
using .Layers: BatchNorm, set_training!, reset_running_stats!, Flatten,
               swish, mish, GlobalAvgPool, DropoutLayer, LayerActivation
using .ConvolutionalLayers: Conv2D, MaxPooling, Conv2DTranspose
using .EmbeddingLayer: Embedding, embedding_forward
using .NeuralNetwork: Sequential, Dense, Activation, collect_parameters,
                      relu, softmax, sigmoid, tanh_activation, leaky_relu,
                      model_to_gpu, model_to_cpu, model_device, model_to_device,
                      layer_to_device, forward
using .Optimizers: SGD, Adam, RMSProp, Adagrad, Nadam, step! as optim_step!
using .Metrics: accuracy, mae, rmse, f1_score, precision, recall, binary_accuracy
using .Reports
using .Training: train!, train_batch!, compute_accuracy_general, train_improved!,
       EarlyStopping, FinalReportCallback, add_callback!,  # Estos ahora vienen de Callbacks
       run_epoch_callbacks, run_final_callbacks, 
       train_with_loaders, stack_batch, evaluate_model, 
       fit!, History
using .Utils: normalize_inputs, set_training_mode!
using .TextUtils: build_vocabulary, text_to_indices, pad_sequence
using .ImageProcessing: load_image, load_images_from_folder, augment_image, prepare_batch
using .PretrainedModels: load_vgg16
using .ConvKernelLayers: ConvKernelLayer
using .DataAugmentation: apply_augmentation, augment_batch
using .LRSchedulers: StepScheduler, CosineAnnealingScheduler, get_lr
using .Models: create_resnet, create_simple_cnn
using .GPUMemoryManager: get_tensor_buffer, check_and_clear_gpu_memory, release_tensor_buffer, clear_cache
using .DataLoaders: DataLoader, optimized_data_loader,  cleanup_data_loader!
using .Callbacks: AbstractCallback, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint,
       PrintCallback, FinalReportCallback,
       on_epoch_begin, on_epoch_end, on_train_begin, on_train_end,
       on_batch_begin, on_batch_end
using .ModelSaver: save_model, load_model, save_checkpoint, load_checkpoint
# ----------------------------------------------------------
# Exportación final
# ----------------------------------------------------------
const f0 = Float32(0)


export Tensor, device_of, same_device, gpu_memory_info, ensure_gpu_memory!, zero_grad!, add, backward, mse_loss,
       Sequential, Dense, forward, collect_parameters,
       relu, sigmoid, tanh_activation, leaky_relu, swish, mish, softmax,
       BatchNorm, Flatten, Reshape, DropoutLayer, LayerActivation, GlobalAvgPool,
       Conv2D, MaxPooling, Conv2DTranspose, ConvKernelLayer,
       train, train_batch!, train!, train_improved!,
       SGD, Adam, RMSProp, Adagrad, Nadam, optim_step!,
       accuracy, mae, rmse, f1_score, precision, recall, binary_accuracy,
       plot_training_progress, plot_metrics, normalize_inputs,
       generate_report, add_callback!, PrintCallback, FinalReportCallback,
       run_epoch_callbacks, run_final_callbacks,
       build_vocabulary, text_to_indices, pad_sequence,
       Embedding, embedding_forward, binary_crossentropy, categorical_crossentropy,
       to_gpu, to_cpu, load_image, load_images_from_folder, augment_image, prepare_batch,
       UNet, load_vgg16, stack_batch,
       model_to_gpu, model_to_cpu, model_device, model_to_device, layer_to_device,
       check_and_clear_gpu_memory, get_tensor_buffer, release_tensor_buffer, clear_cache,
       apply_augmentation, augment_batch,
       StepScheduler, CosineAnnealingScheduler, get_lr,
       create_resnet, create_simple_cnn,
       set_training!, reset_running_stats!, set_training_mode!,DataLoader,  cleanup_data_loader!,
       gpu_available, get_device, set_default_device!, auto_device, validate_gpu_environment, optimized_data_loader,
       Activation,binary_crossentropy_with_logits, fit!, History, evaluate_model, EarlyStopping, compute_accuracy_general,
       AbstractCallback, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint,
       PrintCallback, FinalReportCallback,
       on_epoch_begin, on_epoch_end, on_train_begin, on_train_end,
       on_batch_begin, on_batch_end,f0, save_model, load_model, save_checkpoint, load_checkpoint

end # module
