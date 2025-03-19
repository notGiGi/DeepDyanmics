module KernelOptim

export optimized_conv2d, optimized_conv2d_transpose, fused_conv2d

using NNlib, CUDA

function optimized_conv2d(x, weights; stride, pad, dilation=(1,1))
    if CUDA.has_cuda() && has_cudnn()
        return NNlib.conv(x, weights; stride=stride, pad=pad, dilation=dilation)
    else
        return NNlib.conv(x, weights; stride=stride, pad=pad, dilation=dilation)
    end
end

function optimized_conv2d_transpose(x, weights; stride, pad, dilation=(1,1))
    return NNlib.conv_transpose(x, weights; stride=stride, pad=pad, dilation=dilation)
end

function fused_conv2d(x, weights, bias; stride, pad, dilation=(1,1))
    y = optimized_conv2d(x, weights; stride=stride, pad=pad, dilation=dilation)
    return y .+ reshape(bias, 1, 1, size(bias,1), 1)
end

function has_cudnn()
    try
        using CUDNN
        return true
    catch
        return false
    end
end

end  # module KernelOptim
