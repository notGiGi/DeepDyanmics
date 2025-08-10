############################
# debug_rnn.jl  (copy/paste)
############################

using DeepDynamics
using CUDA
using Random, Statistics
using DeepDynamics.TensorEngine
# ------------ CONFIG -------------
const USE_GPU = CUDA.functional()   # pon false para forzar CPU
Random.seed!(42)

to_dev(x)  = (USE_GPU && CUDA.functional()) ? to_gpu(x) : x
dev_name() = (USE_GPU && CUDA.functional()) ? "GPU" : "CPU"

# --------- helpers: CE desde logits + probs ---------
function ce_from_logits(logits::Tensor, y_true::Tensor)
    z = logits.data; t = y_true.data
    on_gpu = (z isa CUDA.CuArray)
    if on_gpu && !(t isa CUDA.CuArray); t = CUDA.CuArray(t)
    elseif !on_gpu && (t isa CUDA.CuArray); t = Array(t)
    end
    m = maximum(z; dims=1); if on_gpu && !(m isa CUDA.CuArray); m = CUDA.CuArray(m); end
    zst = z .- m
    expz = exp.(zst); sumexp = sum(expz; dims=1); logp = zst .- log.(sumexp)
    B = Float32(size(z,2))
    loss_val = sum(-sum(t .* logp; dims=1)) / B
    result = Tensor(reshape([loss_val], (1,1)); requires_grad=true)
    if logits.requires_grad
        result.backward_fn = grad -> begin
            gs = grad; gs isa Tensor && (gs = gs.data)
            gs isa AbstractArray && (gs = gs[1]); gs = Float32(gs)
            p = expz ./ sumexp
            TensorEngine.backward(logits, Tensor((p .- t) .* (gs / B)))
        end
    end
    return result
end

logits_to_probs(A) = begin
    m = maximum(A; dims=1)
    expz = exp.(A .- m)
    expz ./ sum(expz; dims=1)
end

# --------- helpers: one-hot ---------
# (V,T,B): features=vocab, luego tiempo, luego batch
function to_onehot_VTB(X::AbstractMatrix{<:Integer}, V::Int)
    T, B = size(X)
    OH = zeros(Float32, V, T, B)
    @inbounds for b in 1:B, t in 1:T
        OH[X[t,b], t, b] = 1f0
    end
    return OH
end

# (T,V,B): tiempo, vocab, batch
function to_onehot_TVB(X::AbstractMatrix{<:Integer}, V::Int)
    T, B = size(X)
    OH = zeros(Float32, T, V, B)
    @inbounds for b in 1:B, t in 1:T
        OH[t, X[t,b], b] = 1f0
    end
    return OH
end

# --------- helper: imprimir grad por parámetro ---------
function grad_summary(model; label="")
    println("\n--- Grad summary $label ---")
    ps = collect_parameters(model)
    got = 0
    for (i,p) in enumerate(ps)
        g = p.grad
        has = g !== nothing
        got += has ? 1 : 0
        sz  = size(p.data)
        norm = has ? mean(abs.(g.data)) : NaN
        println(rpad("  p[$i] size=$(sz)", 32), " grad? ", has, has ? "  mean|grad|=$(round(norm,digits=6))" : "")
    end
    println("  ==> Parámetros con grad: $got/$(length(ps))")
    got
end

# --------- helper: labels one-hot (arma en CPU; luego a device) ---------
function make_labels(C, B)
    Y = zeros(Float32, C, B)
    @inbounds for i in 1:B
        Y[(i % C) + 1, i] = 1f0
    end
    to_dev(Tensor(Y))
end

# ======================================
# TEST 1: softmax head (descarta que la cabeza corte grad)
# ======================================
function test_softmax_head(; B=16, inF=8, C=3)
    println("\n[TEST 1 · softmax head · $(dev_name())]")

    # A) Dense -> softmax -> CCE
    modelA = Sequential([ Dense(inF, C), Activation(softmax) ])
    USE_GPU && (modelA = model_to_gpu(modelA))
    x = to_dev(Tensor(randn(Float32, inF, B); requires_grad=true))
    y = make_labels(C, B)

    ypred = forward(modelA, x)
    lossA = categorical_crossentropy(ypred, y)
    zero_grad!(modelA); seed = to_dev(Tensor(ones(Float32,1,1)))
    backward(lossA, seed)
    gA = grad_summary(modelA; label="Dense + softmax + CCE")
    println(gA==0 ? "❌ NO llegan gradientes por softmax" : "✅ Gradientes atraviesan softmax head.")

    # B) Dense (logits) -> CE-from-logits
    modelB = Sequential([ Dense(inF, C) ])
    USE_GPU && (modelB = model_to_gpu(modelB))
    ypredB = forward(modelB, x)
    lossB  = ce_from_logits(ypredB, y)
    zero_grad!(modelB); backward(lossB, seed)
    gB = grad_summary(modelB; label="Dense (logits) + CE-from-logits")

    println((gB>0 && gA==0) ? "➡️  Cambia a 'logits + CE-from-logits' en tu script." :
                              "➡️  Softmax head OK (o ambos OK).")
end

# ======================================
# TEST 2: Embedding en GPU (ver si corta grad)
# ======================================
function test_embedding_chain(; V=10, E=16, H=32, T=5, B=16, C=3)
    println("\n[TEST 2 · Embedding→RNN→Dense · $(dev_name())]")
    model = Sequential([
        Embedding(V, E),
        RNN(E, H; batch_first=false, return_sequences=false, activation=tanh),
        Dense(H, C)   # logits
    ])
    USE_GPU && (model = model_to_gpu(model))

    X = reshape(Int32.(rand(1:V, T*B)), T, B)
    y = make_labels(C, B)

    x = to_dev(Tensor(X))      # Int32 (Embedding)
    logits = forward(model, x)
    loss   = ce_from_logits(logits, y)
    zero_grad!(model); seed = to_dev(Tensor(ones(Float32,1,1)))
    backward(loss, seed)

    got = grad_summary(model; label="Embedding→RNN→Dense (logits)")
    if USE_GPU && got < length(collect_parameters(model))
        println("⚠️ Embedding NO recibe gradiente en GPU. Usa CPU o evita Embedding (TEST 3/4).")
    else
        println("✅ Gradientes completos a través de Embedding en $(dev_name()).")
    end
end

# ======================================
# TEST 3/4: sin Embedding (one-hot) → detectar layout correcto
# Probar:
#   3A) 3D (V,T,B)
#   3B) 3D (T,V,B)
#   4)  2D flatten (V*T, B)  ← suele funcionar con RNN que hace reshape interno
# ======================================
function test_onehot_variants(; V=10, H1=32, H2=32, T=5, B=16, C=3)
    println("\n[TEST 3/4 · one-hot variantes · $(dev_name())]")

    base_model() = Sequential([
        RNN(V,  H1; batch_first=false, return_sequences=true,  activation=tanh),
        RNN(H1, H2; batch_first=false, return_sequences=false, activation=tanh),
        Dense(H2, C)  # logits
    ])

    X = reshape(Int32.(rand(1:V, T*B)), T, B)
    y = make_labels(C, B)

    # ---- 3A: VTB (V,T,B) ----
    modelA = base_model(); USE_GPU && (modelA = model_to_gpu(modelA))
    try
        OH = to_onehot_VTB(X, V)                       # (V,T,B)
        x  = to_dev(Tensor(OH))
        println("   shape VTB: ", size(x.data))
        logits = forward(modelA, x)
        loss   = ce_from_logits(logits, y)
        zero_grad!(modelA); seed = to_dev(Tensor(ones(Float32,1,1)))
        backward(loss, seed)
        gA = grad_summary(modelA; label="one-hot (V,T,B)")
        println(gA>0 ? "✅ Gradientes con (V,T,B)." : "❌ Sin grad con (V,T,B).")
    catch e
        println("❌ (V,T,B) falló: ", sprint(showerror, e))
    end

    # ---- 3B: TVB (T,V,B) ----
    modelB = base_model(); USE_GPU && (modelB = model_to_gpu(modelB))
    try
        OH = to_onehot_TVB(X, V)                       # (T,V,B)
        x  = to_dev(Tensor(OH))
        println("   shape TVB: ", size(x.data))
        logits = forward(modelB, x)
        loss   = ce_from_logits(logits, y)
        zero_grad!(modelB); seed = to_dev(Tensor(ones(Float32,1,1)))
        backward(loss, seed)
        gB = grad_summary(modelB; label="one-hot (T,V,B)")
        println(gB>0 ? "✅ Gradientes con (T,V,B)." : "❌ Sin grad con (T,V,B).")
    catch e
        println("❌ (T,V,B) falló: ", sprint(showerror, e))
    end

    # ---- 4: 2D flatten (V*T, B) ----
    # Muchos RNN de librerías caseras aceptan 2D y hacen reshape interno a (inF, T, B).
    modelC = base_model(); USE_GPU && (modelC = model_to_gpu(modelC))
    try
        OH = to_onehot_VTB(X, V)                       # (V,T,B)
        X2D = reshape(OH, V*T, B)                      # (V*T, B)  ← clave
        x  = to_dev(Tensor(X2D))
        println("   shape flat2D: ", size(x.data))
        logits = forward(modelC, x)
        loss   = ce_from_logits(logits, y)
        zero_grad!(modelC); seed = to_dev(Tensor(ones(Float32,1,1)))
        backward(loss, seed)
        gC = grad_summary(modelC; label="one-hot flatten (V*T, B)")
        println(gC>0 ? "✅ Gradientes con flatten 2D (V*T, B)." : "❌ Sin grad con flatten 2D.")
    catch e
        println("❌ flatten 2D falló: ", sprint(showerror, e))
    end

    println("➡️  Usa el PRIMER layout que marque ✅ arriba (si VTB/TVB fallan, normalmente el flatten 2D pasa).")
end

# ======================================
# RUN ALL
# ======================================
println("=== DEBUG HARNESS (", dev_name(), ") ===")
@time begin
    test_softmax_head()
    test_embedding_chain()
    test_onehot_variants()   # ← incluye 3A, 3B y 4
end
println("\nListo. Observa las líneas con ✅/⚠️/❌ para decidir input layout y si debes evitar Embedding en GPU.")
