module ReshapeModule

using ..TensorEngine
using ..AbstractLayer

export Reshape

# 1. Estructura con el nombre correcto del campo
struct Reshape <: AbstractLayer.Layer
    output_shape::Tuple  # Nombre consistente
end

# 2. Función de forward
function (layer::Reshape)(input::TensorEngine.Tensor)
    input_dims = size(input.data)
    batch_size = input_dims[end]
    
    new_shape = (layer.output_shape..., batch_size)
    new_data = reshape(input.data, new_shape)
    out = TensorEngine.Tensor(new_data)
    
    out.backward_fn = grad -> begin
        grad_input = reshape(grad.data, input_dims)
        TensorEngine.backward(input, TensorEngine.Tensor(grad_input))
    end
    
    return out
end

end  # module ReshapeModule