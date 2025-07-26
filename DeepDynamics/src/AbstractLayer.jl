module AbstractLayer

export Layer, lambda_layer, forward

abstract type Layer end

# Método base forward (opcional)
function forward(layer::Layer, input)
    error("forward not implemented for $(typeof(layer))")
end

# LambdaLayer: envuelve una función para que se comporte como una capa.
struct LambdaLayer <: Layer
    f::Function
end

function lambda_layer(f::Function)
    return LambdaLayer(f)
end

function forward(layer::LambdaLayer, input)
    return layer.f(input)
end

# Hacemos que LambdaLayer sea llamable:
function (layer::LambdaLayer)(input)
    return forward(layer, input)
end



end  # module AbstractLayer
