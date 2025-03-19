module CUDAKernels

using CUDA
export fused_conv_bn_relu!, fused_gemm_relu!

# ==================================================================
# Kernel Fusion: Convolución + BatchNorm + ReLU (Para capas Conv)
# ==================================================================

function fused_conv_bn_relu!(
    output::CuDeviceArray{Float32,4},
    input::CuDeviceArray{Float32,4},
    weights::CuDeviceArray{Float32,4},
    bias::CuDeviceArray{Float32,1},
    gamma::CuDeviceArray{Float32,1},
    beta::CuDeviceArray{Float32,1},
    running_mean::CuDeviceArray{Float32,1},
    running_var::CuDeviceArray{Float32,1},
    epsilon::Float32;
    stride::Int=1,
    padding::Int=1,
    training::Bool=true
)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    batch_size, in_channels, in_height, in_width = size(input)
    out_channels, _, kernel_h, kernel_w = size(weights)

    # Calcular coordenadas de salida
    out_c = ((index - 1) ÷ (batch_size * (in_height ÷ stride) * (in_width ÷ stride))) % out_channels + 1
    b = ((index - 1) ÷ ((in_height ÷ stride) * (in_width ÷ stride))) % batch_size + 1
    out_h = ((index - 1) ÷ (in_width ÷ stride)) % (in_height ÷ stride) + 1
    out_w = (index - 1) % (in_width ÷ stride) + 1

    # Asegurar que estamos dentro de los límites
    if out_c > out_channels || b > batch_size || out_h > (in_height ÷ stride) || out_w > (in_width ÷ stride)
        return
    end

    # Calcular ventana de entrada
    acc = 0.0f0
    for kh in 1:kernel_h
        for kw in 1:kernel_w
            for ic in 1:in_channels
                h = (out_h - 1) * stride + kh - padding
                w = (out_w - 1) * stride + kw - padding
                if 1 <= h <= in_height && 1 <= w <= in_width
                    acc += input[b, ic, h, w] * weights[out_c, ic, kh, kw]
                end
            end
        end
    end

    # Añadir bias y aplicar BatchNorm
    bn_value = gamma[out_c] * (acc + bias[out_c] - running_mean[out_c]) / sqrt(running_var[out_c] + epsilon) + beta[out_c]
    
    # ReLU
    output[index] = training ? max(bn_value, 0.0f0) : bn_value
    return
end

# Wrapper de alto nivel
function fused_conv_bn_relu(
    input::CuArray{Float32,4},
    weights::CuArray{Float32,4},
    bias::CuArray{Float32,1},
    gamma::CuArray{Float32,1},
    beta::CuArray{Float32,1},
    running_mean::CuArray{Float32,1},
    running_var::CuArray{Float32,1};
    epsilon=1e-5f0,
    training=true
)
    out_channels, _, kernel_h, kernel_w = size(weights)
    output_size = (
        size(input, 1),
        out_channels,
        (size(input, 3) + 2*1 - kernel_h) ÷ 1 + 1,
        (size(input, 4) + 2*1 - kernel_w) ÷ 1 + 1
    )
    
    output = CUDA.zeros(Float32, output_size)
    threads = 256
    blocks = ceil(Int, prod(output_size) / threads)

    @cuda threads=threads blocks=blocks fused_conv_bn_relu!(
        output, input, weights, bias, gamma, beta,
        running_mean, running_var, epsilon,
        training=training
    )
    return output
end

# ==================================================================
# Kernel Fusion: GEMM + ReLU (Para capas Densa/FC)
# ==================================================================

function fused_gemm_relu!(output, input, weights, bias, alpha=1.0f0)
    row = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    col = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    if row <= size(output, 1) && col <= size(output, 2)
        acc = 0.0f0
        for k in 1:size(input, 1)
            @inbounds acc += input[k, col] * weights[row, k]
        end
        @inbounds output[row, col] = max(acc * alpha + bias[row], 0.0f0)
    end
    return
end

function fused_gemm_relu(
    input::CuArray{Float32,2},
    weights::CuArray{Float32,2},
    bias::CuArray{Float32,1};
    alpha=1.0f0
)
    output = CUDA.zeros(Float32, size(weights, 1), size(input, 2))
    threads = (32, 32)
    blocks = ceil.(Int, (size(output) ./ threads))

    @cuda threads=threads blocks=blocks fused_gemm_relu!(output, input', weights, bias, alpha)
    return output
end

# ==================================================================
# Funciones de Ayuda
# ==================================================================

function update_running_stats!(
    running_mean::CuArray{Float32,1},
    running_var::CuArray{Float32,1},
    batch_mean::CuArray{Float32,1},
    batch_var::CuArray{Float32,1},
    momentum::Float32
)
    @cuda threads=256 blocks=ceil(Int, length(running_mean)/256) _update_stats_kernel!(
        running_mean, running_var, batch_mean, batch_var, momentum
    )
end

function _update_stats_kernel!(running_mean, running_var, batch_mean, batch_var, momentum)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if idx <= length(running_mean)
        running_mean[idx] = momentum * running_mean[idx] + (1.0f0 - momentum) * batch_mean[idx]
        running_var[idx] = momentum * running_var[idx] + (1.0f0 - momentum) * batch_var[idx]
    end
    return
end

end # module CUDAKernels