module ConvKernelLayers

using CUDA          # para arrays en GPU y programación de kernels
using NNlib         # para convolución en CPU (fallback)
using DeepDynamics  # contexto hipotético (para AbstractLayer, TensorEngine, etc.)
using ..AbstractLayer

export ConvKernelLayer

# Definición del tipo ConvKernelLayer
struct ConvKernelLayer{T<:AbstractFloat, DEV_W, DEV_B} <: AbstractLayer.Layer
    in_channels::Int
    out_channels::Int
    kernel_size::NTuple{2,Int}
    stride::NTuple{2,Int}
    padding::NTuple{2,Int}
    weights::DEV_W   # Array 4D
    bias::DEV_B      # Array 4D con forma (1,1,out_channels,1)
    gradW::DEV_W
    gradB::DEV_B
    # Inner constructor: determina T a partir de los pesos
    function ConvKernelLayer(in_channels::Int, out_channels::Int, kernel_size::NTuple{2,Int},
                             stride::NTuple{2,Int}, padding::NTuple{2,Int},
                             weights, bias, gradW, gradB)
        T = eltype(weights)
        return new{T, typeof(weights), typeof(bias)}(in_channels, out_channels, kernel_size, stride, padding, weights, bias, gradW, gradB)
    end
end

# Constructor externo con opciones de inicialización
function ConvKernelLayer(in_ch::Int, out_ch::Int, kernel_size::NTuple{2,Int};
                         stride::NTuple{2,Int} = (1,1), padding::NTuple{2,Int} = (0,0),
                         initW = :xavier, initB = :zeros)
    kH, kW = kernel_size
    T = Float32  # Tipo de dato fijo

    # Determinar dispositivo: GPU o CPU
    if CUDA.functional()
        DEV_4D = CuArray{T,4}
    else
        DEV_4D = Array{T,4}
    end

    # Inicializar pesos (4D)
    weight_shape = (out_ch, in_ch, kH, kW)
    if initW == :xavier || initW == :glorot
        fan_in = in_ch * kH * kW
        fan_out = out_ch * kH * kW
        limit = sqrt(6.0f0 / (fan_in + fan_out))
        W = rand(T, weight_shape) .* (2f0 * limit) .- limit
        
        W = DEV_4D(W)
    elseif initW == :gaussian
        W = randn(T, weight_shape) .* 0.01f0
        W = DEV_4D(W)
    else  # He initialization por defecto
        fan_in = in_ch * kH * kW
        std = sqrt(2.0f0 / fan_in)
        W = randn(T, weight_shape) .* std
        W = DEV_4D(W)
    end

    # Inicializar bias (1D convertido a 4D)
    if initB == :zeros
        b = zeros(T, out_ch)
    elseif initB == :ones
        b = ones(T, out_ch)
    else
        b = randn(T, out_ch) .* 0.01f0
    end
    b_4D = reshape(b, (1, 1, out_ch, 1))
    b = DEV_4D(b_4D)

    # Inicializar buffers de gradientes
    gradW = similar(W)
    gradB = similar(b)

    return ConvKernelLayer(in_ch, out_ch, kernel_size, stride, padding, W, b, gradW, gradB)
end
# Define constantes globales para el tamaño del tile
# Asegúrate de tener definidas estas constantes globales:


const TILE_H = 16
const TILE_W = 16

function _conv_forward_kernel(d_input, d_weight, d_bias, d_output,
    C_in::Int, H_in::Int, W_in::Int,
    C_out::Int, H_out::Int, W_out::Int,
    kH::Val{KH}, kW::Val{KW}, sH::Val{SH}, sW::Val{SW}, pH::Val{PH}, pW::Val{PW}) where {KH,KW,SH,SW,PH,PW}

    # Calculate output indices
    out_x = (blockIdx().x - 1) * TILE_W + threadIdx().x
    out_y = (blockIdx().y - 1) * TILE_H + threadIdx().y
    idx_z = blockIdx().z - 1
    n = (idx_z ÷ C_out) + 1
    oc = (idx_z % C_out) + 1
    
    # Shared memory with appropriate size
    scratch = @cuStaticSharedMem(Float32, (TILE_H + KH - 1, TILE_W + KW - 1))
    
    # Initialize shared memory to zeros
    ix = threadIdx().x
    iy = threadIdx().y
    
    # Careful bounds checking for shared memory initialization
    if ix <= (TILE_W + KW - 1) && iy <= (TILE_H + KH - 1)
        scratch[iy, ix] = 0.0f0
    end
    
    # Add extra bounds checking for threads that might initialize additional elements
    if ix <= (TILE_W + KW - 1) && (iy + TILE_H) <= (TILE_H + KH - 1)
        scratch[iy + TILE_H, ix] = 0.0f0
    end
    
    if (ix + TILE_W) <= (TILE_W + KW - 1) && iy <= (TILE_H + KH - 1)
        scratch[iy, ix + TILE_W] = 0.0f0
    end
    
    # Ensure all threads have initialized shared memory
    sync_threads()
    
    # Only process valid output locations
    if out_x <= W_out && out_y <= H_out && n <= size(d_input, 1) && oc <= C_out
        # Calculate input coordinates
        in_x_start = (out_x - 1) * SW + 1 - PW
        in_y_start = (out_y - 1) * SH + 1 - PH
        
        val = 0.0f0
        
        # Loop through input channels
        for ic = 1:C_in
            # Load data into shared memory
            # Only threads within the tile boundaries should load data
            if ix <= TILE_W && iy <= TILE_H
                in_x = in_x_start + ix - 1
                in_y = in_y_start + iy - 1
                
                if 1 <= in_x && in_x <= W_in && 1 <= in_y && in_y <= H_in
                    scratch[iy, ix] = d_input[n, ic, in_y, in_x]
                end
            end
            
            # Ensure all threads have loaded data into shared memory
            sync_threads()
            
            # Compute convolution - only for valid output positions
            if out_x <= W_out && out_y <= H_out
                # Manually unroll for 3x3 kernel
                for p = 1:KH
                    for q = 1:KW
                        y_idx = iy + p - 1
                        x_idx = ix + q - 1
                        
                        # Ensure indices are within bounds of shared memory
                        if 1 <= y_idx && y_idx <= (TILE_H + KH - 1) && 
                           1 <= x_idx && x_idx <= (TILE_W + KW - 1)
                            val += d_weight[oc, ic, p, q] * scratch[y_idx, x_idx]
                        end
                    end
                end
            end
            
            # Ensure all threads have finished using the shared memory before next iteration
            sync_threads()
        end
        
        # Write output
        if out_x <= W_out && out_y <= H_out
            d_output[n, oc, out_y, out_x] = val + d_bias[1, 1, oc, 1]
        end
    end
    
    return nothing
end









# Forward pass en GPU: lanza el kernel CUDA
function _gpu_forward!(layer::ConvKernelLayer, x::CuArray{Float32,4})
    # Extract dimensions, with careful order checking
    N, C_in, H_in, W_in = size(x)
    @assert C_in == layer.in_channels "Input channels mismatch"
    
    # Get layer parameters
    kH, kW = layer.kernel_size
    sH, sW = layer.stride
    pH, pW = layer.padding
    
    # Calculate output dimensions
    H_out = div(H_in + 2*pH - kH, sH) + 1
    W_out = div(W_in + 2*pW - kW, sW) + 1
    
    # Create output array with correct dimensions
    y = CUDA.zeros(Float32, N, layer.out_channels, H_out, W_out)
    
    # Calculate grid and block dimensions
    threads_x = min(TILE_W, W_out)
    threads_y = min(TILE_H, H_out)
    threads = (threads_x, threads_y, 1)
    
    # Calculate number of blocks needed
    blocks_x = cld(W_out, threads_x)
    blocks_y = cld(H_out, threads_y)
    blocks_z = N * layer.out_channels
    blocks = (blocks_x, blocks_y, blocks_z)

    
    # Launch kernel with carefully constructed dimensions
    @cuda threads=threads blocks=blocks _conv_forward_kernel(
        x, layer.weights, layer.bias, y,
        C_in, H_in, W_in,
        layer.out_channels, H_out, W_out,
        Val(kH), Val(kW), Val(sH), Val(sW), Val(pH), Val(pW)
    )
    
    return y
end








# Forward pass en CPU: utiliza NNlib.conv
function _cpu_forward!(layer::ConvKernelLayer, x::Array{Float32,4})
    # Verificar formato NCHW
    N, C_in, H_in, W_in = size(x)
    
    if C_in != layer.in_channels
        error("Canales de entrada no coinciden: esperado $(layer.in_channels), recibió $C_in")
    end
    
    # Convertir pesos de (out,in,h,w) a formato NNlib (h,w,in,out)
    W_nnlib = permutedims(layer.weights, (3, 4, 2, 1))
    
    # NNlib conv mantiene formato NCHW
    y = NNlib.conv(x, W_nnlib; stride=layer.stride, pad=layer.padding)
    
    # Añadir bias - ya está en formato (1,1,out,1), convertir a (1,out,1,1)
    bias_reshaped = reshape(layer.bias[1,1,:,1], (1, layer.out_channels, 1, 1))
    y = y .+ bias_reshaped
    
    return y  # Mantiene formato NCHW
end

# Método forward que despacha según el tipo de input
function (layer::ConvKernelLayer)(x::AbstractArray)
    # Añadir información de depuración
    #println("Tipo de datos de entrada: ", typeof(x))
    #println("Dimensiones de entrada: ", size(x))
    
    # Verificar si x es un CuArray
    if x isa CuArray
        #println("Usando GPU forward")
        return _gpu_forward!(layer, x)
    else
        #println("Usando CPU forward")
        return _cpu_forward!(layer, x)
    end
end

function (layer::ConvKernelLayer)(x::DeepDynamics.TensorEngine.Tensor)
    # Asegurarse de que los datos están en GPU si es posible
    if CUDA.functional() && !(x.data isa CuArray)
        #println("Convirtiendo datos a GPU...")
        x_gpu = DeepDynamics.TensorEngine.Tensor(CuArray(x.data))
        result = layer(x_gpu.data)
    else
        result = layer(x.data)
    end
    
    return DeepDynamics.TensorEngine.Tensor(result)
end

end  # module ConvKernelLayers
