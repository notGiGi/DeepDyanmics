module OptimizerLayer

export OptimizerLayer, fuse_layers

using DeepDynamics.AbstractLayer

struct OptimizerLayer{L} <: AbstractLayer.Layer
    layer::L
end

# Definir el constructor una única vez:
OptimizerLayer(layer) = OptimizerLayer{typeof(layer)}(layer)

function forward(opt_layer::OptimizerLayer, input)
    # Aquí se puede agregar lógica para seleccionar implementaciones optimizadas,
    # fusión de operaciones, etc.
    return opt_layer.layer(input)
end

function (opt_layer::OptimizerLayer)(input)
    return forward(opt_layer, input)
end

function fuse_layers(layers::Vector)
    fused = []
    i = 1
    while i <= length(layers)
        if i < length(layers) && layers[i] isa typeof(lambda_layer) && layers[i+1] isa typeof(lambda_layer)
            f1 = layers[i].f
            f2 = layers[i+1].f
            push!(fused, lambda_layer(x -> f2(f1(x))))
            i += 2
        else
            push!(fused, layers[i])
            i += 1
        end
    end
    return fused
end

end  # module OptimizerLayer
