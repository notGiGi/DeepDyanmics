module Metrics

export accuracy, mae, rmse, f1_score, precision, recall, binary_accuracy

function mae(y_pred::Vector{Float64}, y_true::Vector{Float64})
    return mean(abs.(y_pred .- y_true))
end

function rmse(y_pred::Vector{Float64}, y_true::Vector{Float64})
    return sqrt(mean((y_pred .- y_true) .^ 2))
end

function accuracy(y_pred::Union{Vector{Int}, Vector{Float64}}, y_true::Vector{Int}; threshold::Float64=0.5)
    if eltype(y_pred) <: Float64
        y_pred = [p >= threshold ? 1 : 0 for p in y_pred]
    end
    correct = sum(y_pred .== y_true)
    return correct / length(y_true)
end

function precision(y_pred::Vector{Int}, y_true::Vector{Int})
    tp = sum((y_pred .== 1) .& (y_true .== 1))
    fp = sum((y_pred .== 1) .& (y_true .== 0))
    return tp / max(tp + fp, 1e-8)
end

function recall(y_pred::Vector{Int}, y_true::Vector{Int})
    tp = sum((y_pred .== 1) .& (y_true .== 1))
    fn = sum((y_pred .== 0) .& (y_true .== 1))
    return tp / max(tp + fn, 1e-8)
end

function f1_score(y_pred::Vector{Int}, y_true::Vector{Int})
    prec = precision(y_pred, y_true)
    rec = recall(y_pred, y_true)
    return 2 * (prec * rec) / max(prec + rec, 1e-8)
end

function binary_accuracy(y_pred::Vector{Float64}, y_true::Vector{Int}; threshold::Float64=0.5)
    preds = [ p > threshold ? 1 : 0 for p in y_pred ]
    correct = sum(preds .== y_true)
    return correct / length(y_true)
end

end  # module Metrics
