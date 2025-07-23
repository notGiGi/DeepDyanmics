module Models

using ..NeuralNetwork
using ..ConvolutionalLayers
using ..Layers
using ..AbstractLayer
export create_resnet, create_simple_cnn

"""
Crea un modelo ResNet con el número especificado de bloques y clases
"""
function create_resnet(input_channels=3, num_classes=2; blocks=[2,2,2,2])
    layers = AbstractLayer.Layer[]
    
    # Debug
    println("Creating ResNet with input channels: $input_channels")
    
    # Capa de entrada
    initial_channels = 64
    println("Initial conv: $input_channels -> $initial_channels")
    push!(layers, Conv2D(input_channels, initial_channels, (7,7), stride=(2,2), padding=(3,3)))
    push!(layers, BatchNorm(initial_channels))
    push!(layers, LayerActivation(relu))
    push!(layers, MaxPooling((3,3), stride=(2,2), padding=(1,1)))
    
    # Bloques residuales
    in_channels = initial_channels
    for (i, block_count) in enumerate(blocks)
        out_channels = initial_channels * (2^(i-1))
        println("Block section $i: $in_channels -> $out_channels")
        
        stride = i > 1 ? 2 : 1
        println("  First block: stride=$stride")
        push!(layers, create_residual_block(in_channels, out_channels, stride))
        
        for j in 2:block_count
            println("  Block $j: $out_channels -> $out_channels")
            push!(layers, create_residual_block(out_channels, out_channels, 1))
        end
        
        in_channels = out_channels
    end
    
    # Cabeza de clasificación
    println("Final layers: GlobalAvgPool -> $in_channels -> $num_classes")
    push!(layers, GlobalAvgPool())
    push!(layers, Flatten())
    push!(layers, DropoutLayer(Float32(0.5)))
    push!(layers, Dense(in_channels, num_classes))
    push!(layers, LayerActivation(softmax))
    
    return Sequential(layers)
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
