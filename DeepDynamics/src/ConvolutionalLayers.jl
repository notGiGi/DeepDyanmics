module ConvolutionalLayers

using CUDA
using cuDNN                      # Usamos el paquete independiente cuDNN.jl
const CUDNN = cuDNN             # Alias para mayor comodidad
using NNlib, CUDA, ..TensorEngine, ..AbstractLayer
export Conv2D, MaxPooling, Conv2DTranspose

# ---------------------------------------------------------------------------
# Capa Conv2D (usando cuDNN.jl)
# Se asume que:
#   - Los pesos tienen formato (out_ch, in_ch, kH, kW)
#   - La entrada está en formato NCHW: (N, C, H, W)
# ---------------------------------------------------------------------------
struct Conv2D <: AbstractLayer.Layer
    weights::TensorEngine.Tensor    # (out_ch, in_ch, kH, kW)
    bias::TensorEngine.Tensor       # (1, 1, 1, out_ch)
    stride::Tuple{Int,Int}
    padding::Tuple{Int,Int}
    use_batchnorm::Bool
    gamma::Union{TensorEngine.Tensor, Nothing}
    beta::Union{TensorEngine.Tensor, Nothing}
end

function Conv2D(in_ch::Int, out_ch::Int, kernel_size::Tuple{Int,Int};
                stride::Tuple{Int,Int} = (1,1),
                padding::Tuple{Int,Int} = (1,1),
                batchnorm::Bool = false)
    # Filtro en formato cuDNN: (out_ch, in_ch, kH, kW)
    weights = TensorEngine.Tensor(CUDA.randn(Float32, out_ch, in_ch, kernel_size...))
    bias = TensorEngine.Tensor(CUDA.zeros(Float32, 1, 1, 1, out_ch))
    gamma = batchnorm ? TensorEngine.Tensor(CUDA.ones(Float32, 1, out_ch, 1, 1)) : nothing
    beta  = batchnorm ? TensorEngine.Tensor(CUDA.zeros(Float32, 1, out_ch, 1, 1)) : nothing
    return Conv2D(weights, bias, stride, padding, batchnorm, gamma, beta)
end

function (layer::Conv2D)(input::TensorEngine.Tensor)
    x_data = input.data
    if !(x_data isa CUDA.CuArray)
        x_data = CUDA.CuArray(x_data)
    end
    # Convertir parámetros a Int32
    stride_ = Int32[layer.stride...]
    pad_    = Int32[layer.padding...]
    dilation_ = Int32[1, 1]
    
    # Llamar a la función optimizada de cuDNN con parámetros completos
    y = CUDNN.cudnnConvolutionForwardWithDefaults(
            layer.weights.data, 
            x_data;
            stride = stride_,
            pad = pad_,
            dilation = dilation_,
            format = CUDNN.CUDNN_TENSOR_NCHW,
            mode = CUDNN.CUDNN_CROSS_CORRELATION
        )
    
    # Añadir bias
    b = layer.bias.data
    y += reshape(b, (1, size(b, 4), 1, 1))
    return TensorEngine.Tensor(y)
end

# ---------------------------------------------------------------------------
# Capa MaxPooling (usando NNlib.maxpool, que funciona para GPU sin cuDNN)
# ---------------------------------------------------------------------------
struct MaxPooling <: AbstractLayer.Layer
    pool_size::Tuple{Int,Int}
    stride::Tuple{Int,Int}
    padding::Tuple{Int,Int}  # Mantenemos el nombre del campo como padding por compatibilidad
end

# Constructor de MaxPooling - ahora con parámetro de padding
function MaxPooling(pool_size::Tuple{Int,Int}; 
                   stride::Tuple{Int,Int}=pool_size,
                   padding::Tuple{Int,Int}=(0,0))  # Valor por defecto (0,0)
    return MaxPooling(pool_size, stride, padding)
end

# Método forward para MaxPooling
function (layer::MaxPooling)(input::TensorEngine.Tensor)
    x_data = input.data
    # Asegurarse de que x_data es un CuArray (si se usa GPU)
    if !(x_data isa CUDA.CuArray)
        x_data = CUDA.CuArray(x_data)
    end
    
    # Aquí está el cambio clave: usar pad en lugar de padding
    y = NNlib.maxpool(x_data, layer.pool_size; 
                     stride=layer.stride, 
                     pad=layer.padding)  # 'pad' en lugar de 'padding'
    
    return TensorEngine.Tensor(y)
end


# ---------------------------------------------------------------------------
# Capa Conv2DTranspose (usando cuDNN.jl)
# Se asume:
#   - Los pesos tienen formato (in_ch, out_ch, kH, kW)
#   - La entrada está en NCHW: (N, C, H, W)
# ---------------------------------------------------------------------------
struct Conv2DTranspose <: AbstractLayer.Layer
    weights::TensorEngine.Tensor    # (in_ch, out_ch, kH, kW)
    bias::TensorEngine.Tensor       # (1, 1, 1, out_ch)
    stride::Tuple{Int,Int}
    padding::Tuple{Int,Int}
    output_padding::Tuple{Int,Int}
end

function Conv2DTranspose(in_ch::Int, out_ch::Int, kernel_size::Tuple{Int,Int};
                         stride::Tuple{Int,Int} = (1,1),
                         padding::Tuple{Int,Int} = (0,0),
                         output_padding::Tuple{Int,Int} = (0,0))
    weights = TensorEngine.Tensor(randn(Float32, in_ch, out_ch, kernel_size...))
    bias = TensorEngine.Tensor(reshape(zeros(Float32, out_ch), (1,1,1,out_ch)))
    return Conv2DTranspose(weights, bias, stride, padding, output_padding)
end

function (layer::Conv2DTranspose)(input::TensorEngine.Tensor)
    x_data = input.data
    if !(x_data isa CUDA.CuArray)
        x_data = CUDA.CuArray(x_data)
    end
    # Para la conv2D transpuesta usamos la función de cuDNN: 
    # Nota: Asegúrate de que la función (o su equivalente) exista y acepte los argumentos.
    y = CUDNN.cudnnConvolutionForwardWithDefaults(
            layer.weights.data, 
            x_data;
            stride = Int32[layer.stride...],
            pad = Int32[layer.padding...],
            dilation = Int32[1,1],
            format = CUDNN.CUDNN_TENSOR_NCHW,
            mode = CUDNN.CUDNN_CROSS_CORRELATION
        )
    # Añadir bias reestructurado (asegurar que se suma en la dimensión de canales)
    b = layer.bias.data
    y = y .+ reshape(b, (size(y,1), size(b,4), 1, 1))
    return TensorEngine.Tensor(y)
end

function calc_output_size(layer::Conv2DTranspose, x)
    H_in, W_in = size(x)[3:4]
    H_out = (H_in - 1) * layer.stride[1] - 2 * layer.padding[1] + layer.output_padding[1] + size(layer.weights.data, 3)
    W_out = (W_in - 1) * layer.stride[2] - 2 * layer.padding[2] + layer.output_padding[2] + size(layer.weights.data, 4)
    return (H_out, W_out)
end

end # module