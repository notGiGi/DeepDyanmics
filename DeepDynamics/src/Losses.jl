module Losses

using ..TensorEngine
using CUDA
using Statistics

export binary_crossentropy,
       categorical_crossentropy,
       binary_crossentropy_with_logits,
       categorical_crossentropy_from_logits,
       crossentropy_from_logits

# ───────────────────────────────────────────────────────────────────────────────
# Helpers internos
# ───────────────────────────────────────────────────────────────────────────────

# Extrae un escalar 1×1 del gradiente, evitando scalar indexing en GPU
@inline function _scalar(x)
    if x isa Number
        return Float32(x)
    elseif x isa TensorEngine.Tensor
        return _scalar(x.data)
    elseif x isa CUDA.CuArray
        return Float32(Array(x)[1])  # copia mínima y segura
    elseif x isa AbstractArray
        return Float32(x[1])
    else
        return Float32(x)
    end
end

# Asegura que `a` esté en el mismo dispositivo que `ref` (CPU/GPU)
@inline function _to_device_like(a, ref)
    if (ref isa CUDA.CuArray) && !(a isa CUDA.CuArray)
        return CUDA.CuArray(a)
    elseif !(ref isa CUDA.CuArray) && (a isa CUDA.CuArray)
        return Array(a)
    else
        return a
    end
end

# ───────────────────────────────────────────────────────────────────────────────
# 1) BCE sobre probabilidades
# ───────────────────────────────────────────────────────────────────────────────
"""
    binary_crossentropy(y_pred, y_true)

Entropía cruzada binaria suponiendo `y_pred` son probabilidades. CPU/GPU-safe.
Retorna un tensor 1×1 con `requires_grad = true`.
"""
function binary_crossentropy(
    y_pred::TensorEngine.Tensor,
    y_true::TensorEngine.Tensor
)
    # Alinear dispositivo
    is_on_gpu   = (y_pred.data isa CUDA.CuArray)
    y_true_data = _to_device_like(y_true.data, y_pred.data)

    # Forward
    ε = 1f-7
    p_clipped = clamp.(y_pred.data, ε, 1f0 - ε)
    elems = -(
        y_true_data .* log.(p_clipped) .+
        (1f0 .- y_true_data) .* log.(1f0 .- p_clipped)
    )
    N = length(elems)
    loss_val = sum(elems) / N

    # Resultado
    result = TensorEngine.Tensor(reshape([loss_val], (1,1)); requires_grad=true)

    # Backward: (p - y)/N * grad_scalar
    if y_pred.requires_grad
        result.backward_fn = function(grad)
            gs = _scalar(grad)
            grad_input = (y_pred.data .- y_true_data) .* (gs / N)
            TensorEngine.backward(y_pred, TensorEngine.Tensor(grad_input))
        end
    end
    return result
end

# ───────────────────────────────────────────────────────────────────────────────
# 2) BCE desde logits (numéricamente estable)
# ───────────────────────────────────────────────────────────────────────────────
"""
    binary_crossentropy_with_logits(logits, y_true)

Entropía cruzada binaria **desde logits** (estable). CPU/GPU-safe.
Forward:
  L = mean( max(z,0) - z⋅y + log(1 + exp(-|z|)) )
Backward:
  ∂L/∂z = sigmoid(z) - y   (escalado por grad y promedio)
"""
function binary_crossentropy_with_logits(
    logits::TensorEngine.Tensor,
    y_true::TensorEngine.Tensor
)
    # Alinear dispositivo
    t = _to_device_like(y_true.data, logits.data)
    z = logits.data

    # Forward estable
    absz     = abs.(z)
    log_term = log.(1f0 .+ exp.(-absz))
    max_term = clamp.(z, 0f0, Inf)

    loss_elems = max_term .- z .* t .+ log_term
    loss_val   = sum(loss_elems) / length(loss_elems)

    result = TensorEngine.Tensor(reshape([loss_val], (1,1)); requires_grad=true)

    # Backward
    if logits.requires_grad
        result.backward_fn = function(grad)
            gs = _scalar(grad)
            N  = length(z)
            σz = 1f0 ./ (1f0 .+ exp.(-z))
            grad_input = (σz .- t) .* (gs / N)
            TensorEngine.backward(logits, TensorEngine.Tensor(grad_input))
        end
    end
    return result
end

# ───────────────────────────────────────────────────────────────────────────────
# 3) CCE sobre probabilidades
# ───────────────────────────────────────────────────────────────────────────────
"""
    categorical_crossentropy(y_pred, y_true)

Entropía cruzada categórica para multiclase **suponiendo probabilidades** (`softmax` aplicado).
CPU/GPU-safe. Devuelve 1×1 con `requires_grad = true`.
"""
function categorical_crossentropy(
    y_pred::TensorEngine.Tensor,
    y_true::TensorEngine.Tensor
)::TensorEngine.Tensor
    # Dispositivo
    is_on_gpu = (y_pred.data isa CUDA.CuArray)

    # Chequeo de batch
    pred_shape = size(y_pred.data)
    true_shape = size(y_true.data)
    if pred_shape[2] != true_shape[2]
        error("Dimensiones de batch no coinciden: y_pred $(pred_shape), y_true $(true_shape)")
    end

    # Tipos + dispositivo alineados
    y_pred_data = y_pred.data
    y_true_data = _to_device_like(y_true.data, y_pred_data)
    if eltype(y_pred_data) != Float32
        y_pred_data = Float32.(y_pred_data)
    end
    if eltype(y_true_data) != Float32
        y_true_data = Float32.(y_true_data)
    end

    # Forward
    ε = 1.0f-7
    if is_on_gpu
        clipped_pred   = CUDA.clamp.(y_pred_data, ε, 1.0f0 - ε)
        loss_per_batch = -CUDA.sum(y_true_data .* CUDA.log.(clipped_pred), dims=1)
        loss_val       = CUDA.sum(loss_per_batch) / Float32(size(loss_per_batch, 2))
        result = TensorEngine.Tensor(CUDA.reshape([loss_val], (1, 1)); requires_grad=true)
    else
        clipped_pred   = clamp.(y_pred_data, ε, 1.0f0 - ε)
        loss_per_batch = -sum(y_true_data .* log.(clipped_pred), dims=1)
        loss_val       = sum(loss_per_batch) / Float32(size(loss_per_batch, 2))
        result = TensorEngine.Tensor(reshape([loss_val], (1, 1)); requires_grad=true)
    end

    # Backward: (probs - targets) / B * grad_scalar
    if y_pred.requires_grad
        result.backward_fn = grad -> begin
            gs = _scalar(grad)                  # escalar seguro
            B  = Float32(size(y_pred_data, 2))
            scaled_grad = (y_pred_data .- y_true_data) .* (gs / B)
            TensorEngine.backward(y_pred, TensorEngine.Tensor(scaled_grad))
        end
    end
    return result
end

# ───────────────────────────────────────────────────────────────────────────────
# 4) CCE desde logits (softmax + CE en uno, estable)
# ───────────────────────────────────────────────────────────────────────────────
"""
    categorical_crossentropy_from_logits(logits, y_true)

Entropía cruzada categórica **desde logits** (combina log-softmax estable + CE).
CPU/GPU-safe. Devuelve 1×1 con `requires_grad = true`.

- `logits` y `y_true` deben ser de shape (C, B)
- `y_true` puede ser one-hot (Float32/Float64/Int); se convierte a Float32.
"""
function categorical_crossentropy_from_logits(
    logits::TensorEngine.Tensor,
    y_true::TensorEngine.Tensor
)::TensorEngine.Tensor
    # Extraer datos y alinear dispositivo
    z = logits.data                       # (C,B)
    t = _to_device_like(y_true.data, z)   # (C,B)

    # Asegurar Float32 en ambos
    z = (eltype(z) == Float32) ? z : Float32.(z)
    t = (eltype(t) == Float32) ? t : Float32.(t)

    # Chequeo de batch
    size(z,2) == size(t,2) || error("Batch mismatch: logits $(size(z)), targets $(size(t))")

    # Log-softmax estable
    m          = maximum(z; dims=1)                   # (1,B)
    m          = _to_device_like(m, z)
    zst        = z .- m                               # (C,B)
    expz       = exp.(zst)                            # (C,B)
    sumexp     = sum(expz; dims=1)                    # (1,B)
    logsumexp  = log.(sumexp)                         # (1,B)
    logp       = zst .- logsumexp                     # (C,B)

    # Loss promedio por batch
    B         = Float32(size(z, 2))
    ce_per    = -sum(t .* logp; dims=1)               # (1,B)
    loss_val  = sum(ce_per) / B

    # Resultado
    result = TensorEngine.Tensor(reshape([loss_val], (1,1)); requires_grad=true)

    # Backward: ∂L/∂z = (softmax(z) - t)/B * grad_scalar
    if logits.requires_grad
        result.backward_fn = function(grad)
            gs = _scalar(grad)
            p  = expz ./ sumexp                        # (C,B)
            grad_z = (p .- t) .* (gs / B)
            TensorEngine.backward(logits, TensorEngine.Tensor(grad_z))
        end
    end
    return result
end

# Alias conveniente
const crossentropy_from_logits = categorical_crossentropy_from_logits

end # module
