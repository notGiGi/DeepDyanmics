module ConvolutionalLayers

using CUDA
using cuDNN                      # Usamos el paquete independiente cuDNN.jl
const CUDNN = cuDNN             # Alias para mayor comodidad
using NNlib, CUDA, ..TensorEngine, ..AbstractLayer
export Conv2D, MaxPooling, Conv2DTranspose

# -----------------------------------------------------------------------
# Conv2D: constructor + forward (reparación nombre‑arg y padding 0)
# -----------------------------------------------------------------------
# ---------------------------------------------------------------------
#  Conv2D (NCHW)  – con backward usando NNlib.conv_filter / conv_data
# ---------------------------------------------------------------------
struct Conv2D <: AbstractLayer.Layer
    weights::TensorEngine.Tensor        # (out_ch, in_ch, kH, kW)
    bias   ::TensorEngine.Tensor        # (1, out_ch, 1, 1)
    stride ::Tuple{Int,Int}
    padding::Tuple{Int,Int}
    use_batchnorm::Bool
    gamma  ::Union{TensorEngine.Tensor,Nothing}
    beta   ::Union{TensorEngine.Tensor,Nothing}
end

"""
    Conv2D(in_ch, out_ch, (kH,kW); stride=(1,1), padding=(0,0), batchnorm=false)

Capa convolucional 2‑D formato **NCHW** con inicialización He‑normal.
Los parámetros `weights` y `bias` se crean con `requires_grad=true`.
Si `batchnorm==true` agrega los tensores `gamma`/`beta`, también
entrenables, aunque normalmente se usa un `BatchNorm` aparte.
"""
function Conv2D(in_ch::Int, out_ch::Int, ks::Tuple{Int,Int};
                stride::Tuple{Int,Int}=(1,1),
                padding::Tuple{Int,Int}=(0,0),
                batchnorm::Bool=false)

    kH,kW = ks
    w = randn(Float32,out_ch,in_ch,kH,kW) *
        √(2f0/(in_ch*kH*kW))                 # He init
    b = zeros(Float32,1,out_ch,1,1)

    weights = TensorEngine.Tensor(w; requires_grad=true)
    bias    = TensorEngine.Tensor(b; requires_grad=true)

    γ = batchnorm ? TensorEngine.Tensor(ones(Float32,1,out_ch,1,1);
                                        requires_grad=true)  : nothing
    β = batchnorm ? TensorEngine.Tensor(zeros(Float32,1,out_ch,1,1);
                                        requires_grad=true)  : nothing

    Conv2D(weights, bias, stride, padding, batchnorm, γ, β)
end



function (layer::Conv2D)(input::TensorEngine.Tensor)
    x = input.data                                 # (N,C,H,W)

    if ndims(x) != 4
        throw(ErrorException(
            "Conv2D espera tensor 4‑D (N,C,H,W), recibió $(ndims(x))‑D"))
    end

    # ‑‑ mover parámetros al mismo dispositivo que la entrada
    on_gpu = x isa CUDA.CuArray
    if on_gpu && !(layer.weights.data isa CUDA.CuArray)
        layer.weights.data = CUDA.CuArray(layer.weights.data)
        layer.bias.data    = CUDA.CuArray(layer.bias.data)
    elseif !on_gpu && (layer.weights.data isa CUDA.CuArray)
        layer.weights.data = Array(layer.weights.data)
        layer.bias.data    = Array(layer.bias.data)
    end

    # Formatos que NNlib espera
    x_hwcn = permutedims(x, (3,4,2,1))                  # H,W,Cin,N
    w_hwio = permutedims(layer.weights.data, (3,4,2,1)) # kH,kW,Cin,Cout

    y_hwcn = NNlib.conv(x_hwcn, w_hwio;
                        stride=layer.stride,
                        pad   =layer.padding)

    y_hwcn .+= reshape(layer.bias.data, (1,1,size(w_hwio,4),1))
    y = permutedims(y_hwcn, (4,3,1,2))                  # N,C,H,W

    # ----------------------   backward closure   ----------------------
    needs_grad = input.requires_grad ||
                 layer.weights.requires_grad ||
                 layer.bias.requires_grad

    out = TensorEngine.Tensor(y; requires_grad=needs_grad)

    if needs_grad
    # capturamos variables para backward
    stride = layer.stride; pad = layer.padding
    kH, kW = size(layer.weights.data, 3), size(layer.weights.data, 4)
    
    # AGREGAR: Crear ConvDims para los gradientes
    # Necesitamos las dimensiones de entrada
    H_in = size(x_hwcn, 1)
    W_in = size(x_hwcn, 2)
    cdims = NNlib.DenseConvDims(x_hwcn, w_hwio;
                            stride=stride, padding=pad, dilation=(1,1))
    
    out.backward_fn = grad -> begin
        dy = grad isa TensorEngine.Tensor ? grad.data : grad
        dy_hwcn = permutedims(on_gpu ? dy : Array(dy), (3,4,2,1))
        
        # grad W - USAR cdims
        if layer.weights.requires_grad
            g_w_hwio = try
                NNlib.∇conv_filter(x_hwcn, dy_hwcn, cdims)  # Pasar cdims
            catch
                # Fallback manual...
                    # Fallback: calcular manualmente el gradiente de pesos
                    # Para cada peso, su gradiente es la correlación entre
                    # la entrada y el gradiente de salida
                    
                    N = size(x, 1)
                    kH, kW = size(layer.weights.data, 3), size(layer.weights.data, 4)
                    C_in = size(x, 2)
                    C_out = size(dy, 2)
                    
                    # Inicializar gradiente de pesos
                    g_w_hwio = zeros(Float32, kH, kW, C_in, C_out)
                    
                    # Para cada ejemplo del batch
                    for n in 1:N
                        # Extraer el ejemplo n
                        x_n = x_hwcn[:, :, :, n]  # (H, W, C_in)
                        dy_n = dy_hwcn[:, :, :, n]  # (H_out, W_out, C_out)
                        
                        # Calcular correlación para este ejemplo
                        # Necesitamos "deshacer" la convolución
                        for c_out in 1:C_out
                            for c_in in 1:C_in
                                # Canal de entrada y gradiente de salida
                                x_chan = x_n[:, :, c_in]
                                dy_chan = dy_n[:, :, c_out]
                                
                                # Correlación 2D
                                for kh in 1:kH, kw in 1:kW
                                    sum_grad = 0.0f0
                                    
                                    # Recorrer todas las posiciones de salida
                                    H_out, W_out = size(dy_chan)
                                    for h_out in 1:H_out, w_out in 1:W_out
                                        # Posición correspondiente en la entrada
                                        h_in = (h_out - 1) * stride[1] + kh - pad[1]
                                        w_in = (w_out - 1) * stride[2] + kw - pad[2]
                                        
                                        # Verificar límites
                                        if h_in >= 1 && h_in <= size(x_chan, 1) &&
                                           w_in >= 1 && w_in <= size(x_chan, 2)
                                            sum_grad += x_chan[h_in, w_in] * dy_chan[h_out, w_out]
                                        end
                                    end
                                    
                                    g_w_hwio[kh, kw, c_in, c_out] += sum_grad
                                end
                            end
                        end
                    end
                    
                    g_w_hwio
                end
                
                g_w = permutedims(g_w_hwio, (4,3,1,2))                 # (out,in,kH,kW)
                TensorEngine.backward(layer.weights,
                                      TensorEngine.Tensor(on_gpu ? CUDA.CuArray(g_w) : g_w;
                                                          requires_grad=false))
            end

            # grad bias
            if layer.bias.requires_grad
                g_b = sum(dy, dims=(1,3,4))
                TensorEngine.backward(layer.bias,
                                      TensorEngine.Tensor(on_gpu ? CUDA.CuArray(g_b) : g_b;
                                                          requires_grad=false))
            end

            # grad input - TAMBIÉN CORREGIR conv_data
            if input.requires_grad
                g_x_hwcn = try
                    NNlib.∇conv_data(dy_hwcn, w_hwio, cdims)  # Pasar cdims
                catch
                    # Fallback: usar convolución transpuesta
                    # Rotar 180° los pesos
                    w_flipped = w_hwio[end:-1:1, end:-1:1, :, :]
                    
                    # Padding para convolución transpuesta
                    # Este cálculo es complejo y depende de stride/padding originales
                    out_pad = (kH - pad[1] - 1, kW - pad[2] - 1)
                    
                    # Convolución transpuesta
                    NNlib.conv(dy_hwcn, w_flipped;
                              stride=(1,1),
                              pad=out_pad,
                              dilation=(1,1))
                end
                
                g_x = permutedims(g_x_hwcn, (4,3,1,2))
                TensorEngine.backward(input,
                                      TensorEngine.Tensor(on_gpu ? CUDA.CuArray(g_x) : g_x;
                                                          requires_grad=false))
            end
        end
    end
    return out
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



###############################################
#   forward / backward
###############################################
function (layer::MaxPooling)(input::TensorEngine.Tensor)

    ## -------- forward --------
    x_nchw = input.data                       # (N,C,H,W) — conv estándar
    x_hwcn = permutedims(x_nchw, (3,4,2,1))   # (H,W,C,N) — formato NNlib

    y_hwcn = NNlib.maxpool(
        x_hwcn, layer.pool_size;
        stride = layer.stride,
        pad    = layer.padding
    )

    y_nchw = permutedims(y_hwcn, (4,3,1,2))   # de vuelta a (N,C,H',W')

    out = TensorEngine.Tensor(
        y_nchw;
        requires_grad = input.requires_grad       # ⧉ propaga flag
    )

    ## -------- backward --------
    if out.requires_grad
        # capturamos cosas que la closure necesita
        pool   = layer.pool_size
        stride = layer.stride
        pad    = layer.padding

        out.backward_fn = Δ -> begin
            Δ_nchw = Δ isa TensorEngine.Tensor ? Δ.data : Δ
            Δ_hwcn = permutedims(Δ_nchw, (3,4,2,1))

            # 1️⃣  intento rápido: usar API de NNlib si existe
            grad_x_hwcn = try
                NNlib.∇maxpool(Δ_hwcn, y_hwcn, x_hwcn, pool;
                               stride = stride, pad = pad)
            catch
                # 2️⃣  fallback: máscara del máximo (CPU) --------------
                grad_cpu = zeros(Float32, size(x_hwcn))
                H_out, W_out, C, N = size(y_hwcn)
                ph, pw = pool
                sh, sw = stride
                for n in 1:N, c in 1:C, h in 1:H_out, w in 1:W_out
                    h0 = (h-1)*sh + 1
                    w0 = (w-1)*sw + 1
                    h1 = min(h0+ph-1, size(x_hwcn,1))
                    w1 = min(w0+pw-1, size(x_hwcn,2))
                    window = @view x_hwcn[h0:h1, w0:w1, c, n]
                    # pos máx local
                    maxidx = findmax(window)[2]
                    grad_cpu[h0:h1, w0:w1, c, n][maxidx] += Δ_hwcn[h,w,c,n]
                end
                # si input estaba en GPU, subimos el gradiente
                x_hwcn isa CUDA.CuArray ? CUDA.CuArray(grad_cpu) : grad_cpu
            end

            grad_x_nchw = permutedims(grad_x_hwcn, (4,3,1,2))

            TensorEngine.backward(
                input,
                TensorEngine.Tensor(
                    TensorEngine.ensure_compatible(grad_x_nchw, input.data);
                    requires_grad = false
                )
            )
        end
    end

    return out
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