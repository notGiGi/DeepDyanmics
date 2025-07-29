module Metrics
using CUDA
export accuracy, mae, rmse, f1_score, precision, recall, binary_accuracy

function mae(y_pred::Vector{Float64}, y_true::Vector{Float64})
    return mean(abs.(y_pred .- y_true))
end

function rmse(y_pred::Vector{Float64}, y_true::Vector{Float64})
    return sqrt(mean((y_pred .- y_true) .^ 2))
end

# Si accuracy espera vectores, agregar conversión:
function accuracy(y_pred, y_true; threshold::Float64=0.5)
    # Convertir a arrays si son CuArrays
    y_pred_cpu = y_pred isa CUDA.CuArray ? Array(y_pred) : y_pred
    y_true_cpu = y_true isa CUDA.CuArray ? Array(y_true) : y_true
    
    # Si son matrices (multiclase), usar argmax
    if ndims(y_pred_cpu) == 2 && ndims(y_true_cpu) == 2
        pred_classes = vec(argmax(y_pred_cpu, dims=1))
        true_classes = vec(argmax(y_true_cpu, dims=1))
        return sum(pred_classes .== true_classes) / length(pred_classes)
    
    # Si y_pred es matriz pero y_true es vector
    elseif ndims(y_pred_cpu) == 2 && ndims(y_true_cpu) == 1
        pred_classes = vec(argmax(y_pred_cpu, dims=1))
        return sum(pred_classes .== y_true_cpu) / length(y_true_cpu)
    
    # Caso vectoress
    else
        y_pred_vec = vec(y_pred_cpu)
        y_true_vec = vec(y_true_cpu)
        if eltype(y_pred_vec) <: AbstractFloat
            y_pred_vec = [p >= threshold ? 1 : 0 for p in y_pred_vec]
        end
        correct = sum(y_pred_vec .== y_true_vec)
        return correct / length(y_true_vec)
    end
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

function binary_accuracy(y_pred, y_true; threshold::Float64=0.5)
    # Convertir a arrays si son CuArrays
    y_pred_cpu = y_pred isa CUDA.CuArray ? Array(y_pred) : y_pred
    y_true_cpu = y_true isa CUDA.CuArray ? Array(y_true) : y_true
    
    # Aplanar a vectores
    y_pred_vec = vec(y_pred_cpu)
    y_true_vec = vec(y_true_cpu)
    
    # Aplicar threshold
    preds = [p > threshold ? 1 : 0 for p in y_pred_vec]
    
    # Si y_true es float, convertir a int
    if eltype(y_true_vec) <: AbstractFloat
        y_true_vec = [y > threshold ? 1 : 0 for y in y_true_vec]
    end
    
    correct = sum(preds .== y_true_vec)
    return correct / length(y_true_vec)
end

end  # module Metrics
