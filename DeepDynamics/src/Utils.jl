module Utils
using ..NeuralNetwork
using ..TensorEngine
using Statistics
using ..Layers
export normalize_inputs, set_training_mode!

function normalize_inputs(inputs::Vector{TensorEngine.Tensor})
    all_data = hcat([input.data for input in inputs]...)
    mean_val = Statistics.mean(all_data, dims=2)
    std_val  = Statistics.std(all_data, dims=2) .+ 1e-8
    return [TensorEngine.Tensor((input.data .- mean_val) ./ std_val) for input in inputs]
end

"""
    set_training_mode!(model, training::Bool)

Establece el modo training/eval para todas las capas del modelo.
Función unificada definida solo en el módulo principal.
"""

function set_training_mode!(model, training::Bool)
    if isa(model, NeuralNetwork.Sequential)
        for layer in model.layers
            set_training_mode!(layer, training)
        end
    elseif isa(model, Layers.BatchNorm) || isa(model, Layers.DropoutLayer) || 
           isa(model, Layers.LayerNorm)
        model.training = training
    elseif isa(model, Layers.RNNCell) || isa(model, Layers.LSTMCell) || 
           isa(model, Layers.GRUCell)
        model.training = training
    elseif isa(model, Layers.RNN) || isa(model, Layers.LSTM) || 
           isa(model, Layers.GRU)
        model.cell.training = training
    end
    return model
end

end  # module Utils
