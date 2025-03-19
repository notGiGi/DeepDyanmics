module AutoTuner

export tune_kernel_params, auto_tune_conv

using BenchmarkTools, NNlib, CUDA

function tune_kernel_params(x, weights; strides, pads, dilations=[(1,1)])
    best_time = Inf
    best_config = nothing
    for stride in strides
        for pad in pads
            for dilation in dilations
                t = @belapsed NNlib.conv($x, $weights; stride=$stride, pad=$pad, dilation=$dilation)
                if t < best_time
                    best_time = t
                    best_config = (stride, pad, dilation)
                end
            end
        end
    end
    return best_config, best_time
end

function auto_tune_conv(x, weights)
    strides = [(1,1), (2,2)]
    pads = [(0,0), (1,1)]
    best_config, best_time = tune_kernel_params(x, weights; strides=strides, pads=pads)
    @info "Auto-tuning: Mejor configuración: $best_config, tiempo: $best_time seg"
    return best_config
end

end  # module AutoTuner
