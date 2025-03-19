module Utils

using ..TensorEngine
using Statistics

export normalize_inputs

function normalize_inputs(inputs::Vector{TensorEngine.Tensor})
    all_data = hcat([input.data for input in inputs]...)
    mean_val = Statistics.mean(all_data, dims=2)
    std_val  = Statistics.std(all_data, dims=2) .+ 1e-8
    return [TensorEngine.Tensor((input.data .- mean_val) ./ std_val) for input in inputs]
end

end  # module Utils
