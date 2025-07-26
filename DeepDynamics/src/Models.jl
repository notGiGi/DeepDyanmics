module Models

using ..NeuralNetwork
using ..ConvolutionalLayers
using ..Layers
using ..AbstractLayer
export create_resnet, create_simple_cnn

"""
Crea un modelo ResNet con el número especificado de bloques y clases
"""
function create_resnet(input_channels::Int=3, num_classes::Int=2; 
                      blocks::Vector{Int}=[2,2,2,2], input_size::Int=224)
    
    layers = AbstractLayer.Layer[]
    
    # Validar entradas
    @assert input_channels > 0 "input_channels debe ser positivo"
    @assert num_classes > 0 "num_classes debe ser positivo"
    @assert all(b > 0 for b in blocks) "Todos los bloques deben ser positivos"
    @assert input_size > 0 "input_size debe ser positivo"
    
    # Rastrear dimensiones espaciales
    current_spatial_dim = input_size
    
    # Capa inicial
    initial_channels = 64
    push!(layers, Conv2D(input_channels, initial_channels, (7,7), 
                        stride=(2,2), padding=(3,3)))
    current_spatial_dim = compute_conv_output_dim(current_spatial_dim, 7, 2, 3)
    
    push!(layers, BatchNorm(initial_channels))
    push!(layers, LayerActivation(relu))
    
    # MaxPooling
    push!(layers, MaxPooling((3,3), stride=(2,2), padding=(1,1)))
    current_spatial_dim = div(current_spatial_dim + 2*1 - 3, 2) + 1
    
    # Bloques residuales
    in_channels = initial_channels
    
    for (stage_idx, block_count) in enumerate(blocks)
        out_channels = initial_channels * (2^(stage_idx-1))
        
        # Primer bloque de cada etapa puede reducir dimensiones
        stride = stage_idx > 1 ? 2 : 1
        
        # Crear primer bloque
        push!(layers, create_residual_block(in_channels, out_channels, stride))
        
        if stride == 2
            current_spatial_dim = compute_conv_output_dim(current_spatial_dim, 3, stride, 1)
        end
        
        # Resto de bloques mantienen dimensiones
        for _ in 2:block_count
            push!(layers, create_residual_block(out_channels, out_channels, 1))
        end
        
        in_channels = out_channels
    end
    
    # Pooling global adaptativo
    push!(layers, GlobalAvgPool())
    push!(layers, Flatten())
    
    # Capas finales
    push!(layers, DropoutLayer(Float32(0.5)))
    push!(layers, Dense(in_channels, num_classes))
    push!(layers, LayerActivation(softmax))
    
    return Sequential(layers)
end

# Helper function
function compute_conv_output_dim(input_dim::Int, kernel_size::Int, 
                                stride::Int, padding::Int)
    return div(input_dim + 2*padding - kernel_size, stride) + 1
end

"""
Crea un modelo CNN simple para clasificación
"""
function create_simple_cnn(input_channels=3, num_classes=2)
    return Sequential([
        Conv2D(input_channels, 32, (3,3), stride=(1,1), padding=(1,1)),
        BatchNorm(32),
        NeuralNetwork.Activation(relu),
        MaxPooling((2,2)),
        
        Conv2D(32, 64, (3,3), stride=(1,1), padding=(1,1)),
        BatchNorm(64),
        NeuralNetwork.Activation(relu),
        MaxPooling((2,2)),
        
        Flatten(),
        Dense(64 * 56 * 56, 128),
        BatchNorm(128),
        NeuralNetwork.Activation(relu),
        DropoutLayer(0.5),
        Dense(128, num_classes),
        NeuralNetwork.Activation(softmax)
    ])
end

end # module Models
