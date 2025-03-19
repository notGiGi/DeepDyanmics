module Models

using ..NeuralNetwork
using ..ConvKernelLayers
using ..Layers
using ..ConvolutionalLayers
using ..AbstractLayer
export create_resnet, create_simple_cnn

"""
Crea un modelo ResNet con el número especificado de bloques y clases
"""
# En Models.jl - Versión corregida de create_resnet con debugging
function create_resnet(input_channels=3, num_classes=2; blocks=[2,2,2,2])
    layers = AbstractLayer.Layer[]
    
    # Imprimimos información de depuración
    println("Creating ResNet with input channels: $input_channels")
    
    # Capa de entrada - siempre debe tener el número correcto de canales de entrada
    initial_channels = 64
    println("Initial conv: $input_channels -> $initial_channels")
    push!(layers, ConvKernelLayer(input_channels, initial_channels, (7,7), stride=(2,2), padding=(3,3)))
    push!(layers, BatchNorm(initial_channels))
    push!(layers, LayerActivation(relu))
    push!(layers, MaxPooling((3,3), stride=(2,2), padding=(1,1)))
    
    # Bloques residuales - hacemos un seguimiento de los canales entre bloques
    in_channels = initial_channels
    for (i, block_count) in enumerate(blocks)
        out_channels = initial_channels * (2^(i-1))
        
        println("Block section $i: $in_channels -> $out_channels")
        
        # Primer bloque puede tener stride para reducir dimensiones
        stride = i > 1 ? 2 : 1
        println("  First block: stride=$stride")
        push!(layers, create_residual_block(in_channels, out_channels, stride))
        
        # Resto de bloques en esta sección mantienen dimensiones
        for j in 2:block_count
            println("  Block $j: $out_channels -> $out_channels")
            push!(layers, create_residual_block(out_channels, out_channels, 1))
        end
        
        # Actualizar canales para la siguiente sección
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
        ConvKernelLayer(input_channels, 32, (3,3), stride=(1,1), padding=(1,1)),
        BatchNorm(32),
        NeuralNetwork.Activation(relu),
        MaxPooling((2,2)),
        
        ConvKernelLayer(32, 64, (3,3), stride=(1,1), padding=(1,1)),
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