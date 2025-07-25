

module DeepDynamics
#using Base.Threads
# Incluir los submódulos en el orden adecuado:
include("GPUMemoryManager.jl")
include("DataAugmentation.jl")
include("AbstractLayer.jl")
include("TensorEngine.jl")

include("ConvKernelLayers.jl")
include("Visualizations.jl")
include("ReshapeModule.jl")
include("Losses.jl")
include("ConvolutionalLayers.jl")
include("Layers.jl")
include("EmbeddingLayer.jl")
include("NeuralNetwork.jl")
include("Utils.jl")
include("Optimizers.jl")
include("Metrics.jl")
include("Reports.jl")
include("DataLoaders.jl")
include("Training.jl")

include("TextUtils.jl")
include("ImageProcessing.jl")
include("UNetSegmentation.jl")
include("PretrainedModels.jl")          # <<-- Ahora Training puede usar ProfilingTools
include("LRSchedulers.jl")
include("Models.jl")


#include("ProfilingTools.jl")
# Importar y reexportar símbolos clave:
using .TensorEngine: Tensor, backward, mse_loss, 
       initialize_grad!, initialize_weights, l2_regularization,
       compute_loss_with_regularization, clip_gradients!, 
       to_gpu, to_cpu, zero_grad!, add, device_of, same_device, gpu_memory_info, ensure_gpu_memory!
using .Visualizations
using .ReshapeModule: Reshape
using .Losses: binary_crossentropy, categorical_crossentropy
using .Layers:  BatchNorm, set_training!, reset_running_stats!, Flatten, swish, mish, GlobalAvgPool, DropoutLayer, LayerActivation
using .ConvolutionalLayers: Conv2D, MaxPooling, Conv2DTranspose
using .EmbeddingLayer: Embedding, embedding_forward
using .NeuralNetwork: Sequential, Dense, Activation, collect_parameters,
       relu, softmax, sigmoid, tanh_activation, leaky_relu,
       model_to_gpu, model_to_cpu, model_device, model_to_device,
       layer_to_device, forward
using .Optimizers: SGD, Adam, RMSProp, Adagrad, Nadam
using .Optimizers: step! as optim_step!  # Renombrar para evitar conflicto
using .Metrics: accuracy, mae, rmse, f1_score, precision, recall, binary_accuracy
using .Reports
using .Training: train_batch!, stack_batch, train_improved!,  train!, EarlyStopping, Callback, PrintCallback, FinalReportCallback, add_callback!, run_epoch_callbacks, run_final_callbacks
using .Utils: normalize_inputs, set_training_mode!
using .TextUtils: build_vocabulary, text_to_indices, pad_sequence
using .ImageProcessing: load_image, load_images_from_folder, augment_image, prepare_batch
#using .UNetSegmentation: UNet, forward
using .PretrainedModels: load_vgg16
using .ConvKernelLayers: ConvKernelLayer
using .DataAugmentation: apply_augmentation, augment_batch
using .LRSchedulers: StepScheduler, CosineAnnealingScheduler, get_lr
using .Models: create_resnet, create_simple_cnn
# using .ProfilingTools: @profile_block, print_profile_summary  # Ignorado por ahora
using .GPUMemoryManager: get_tensor_buffer, check_and_clear_gpu_memory, release_tensor_buffer, clear_cache
using .DataLoaders: DataLoader, optimized_data_loader



export Tensor,device_of, same_device, gpu_memory_info, ensure_gpu_memory!,
       Sequential, Dense, forward, collect_parameters, relu, sigmoid, tanh_activation, leaky_relu,
       BatchNorm, Flatten, Reshape,
       Conv2D, MaxPooling, Conv2DTranspose,
       train, train_batch!, train!,
       SGD, Adam, RMSProp, Adagrad, Nadam, step!,
       accuracy, mae, rmse, f1_score, precision, recall, binary_accuracy,
       plot_training_progress, plot_metrics, normalize_inputs,
       generate_report, add_callback!, PrintCallback, FinalReportCallback, run_epoch_callbacks, run_final_callbacks,
       compute_accuracy_general,
       build_vocabulary, text_to_indices, pad_sequence,
       Embedding, embedding_forward, binary_crossentropy, to_cpu, to_gpu, softmax, categorical_crossentropy,
       load_image, load_images_from_folder, augment_image, prepare_batch,
       UNet, load_vgg16, stack_batch, Activation, optim_step!, ConvKernelLayer, apply_augmentation, 
       augment_batch, StepScheduler, CosineAnnealingScheduler, get_lr, train_improved!, DropoutLayer,
       create_resnet, create_simple_cnn, LayerActivation, apply_augmentation, augment_batch,
       get_tensor_buffer, release_tensor_buffer, clear_cache, DataLoader, optimized_data_loader, zero_grad!, add, backward, mse_loss,
       set_training!, reset_running_stats!, model_to_gpu, model_to_cpu, model_device, model_to_device,
       layer_to_device, check_and_clear_gpu_memory, set_training_mode!




end  # module DeepDynamics