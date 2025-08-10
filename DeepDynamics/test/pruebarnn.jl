# test/pruebarnn.jl
using DeepDynamics
using CUDA
using Random
using Statistics
using DeepDynamics.Optimizers: step!
using DeepDynamics.Losses: crossentropy_from_logits

function generate_patterned_data(n_samples; seq_len=10, vocab_size=10, n_classes=3)
    X = zeros(Int32, seq_len, n_samples)
    Y = zeros(Float32, n_classes, n_samples)
    for i in 1:n_samples
        class = rand(0:2)
        seq = Int32[]
        for _ in 1:seq_len
            if rand() < 0.7
                if class == 0
                    push!(seq, rand(1:3))
                elseif class == 1
                    push!(seq, rand(4:6))
                else
                    push!(seq, rand(7:10))
                end
            else
                push!(seq, rand(1:vocab_size))
            end
        end
        X[:, i] = seq
        Y[class+1, i] = 1.0f0
    end
    return X, Y
end

# One-hot: (T,B) índices → (T,B,V) Float32
function to_onehot_TBV(X::AbstractMatrix{<:Integer}, V::Int)
    T, B = size(X)
    OH = zeros(Float32, T, B, V)
    @inbounds for b in 1:B, t in 1:T
        OH[t, b, X[t,b]] = 1f0
    end
    return OH
end

# Softmax por columnas (C,B) → (C,B)
softmax_cols(A) = begin
    m = maximum(A; dims=1)
    expz = exp.(A .- m)
    expz ./ sum(expz; dims=1)
end

# Batching simple
function create_batches(X, Y, batch_size; shuffle=true)
    n = size(X, 2)
    idxs = shuffle ? Random.shuffle(1:n) : (1:n)
    batches = Vector{Tuple{Matrix{Int32}, Matrix{Float32}}}()
    for i in 1:batch_size:n
        j = min(i + batch_size - 1, n)
        push!(batches, (X[:, idxs[i:j]], Y[:, idxs[i:j]]))
    end
    return batches
end

function main()
    Random.seed!(42)
    println("=== RNN para Clasificación de Patrones en Secuencias (GPU, logits CE) — limpio ===\n")

    # ----------------------
    # Datos
    # ----------------------
    n_train    = 5000
    n_test     = 1000
    seq_len    = 10
    vocab_size = 10
    n_classes  = 3

    println("📊 Generando datos con patrón detectable...")
    X_train, y_train = generate_patterned_data(n_train; seq_len, vocab_size, n_classes)
    X_test,  y_test  = generate_patterned_data(n_test;  seq_len, vocab_size, n_classes)

    train_dist = vec(sum(y_train, dims=2))
    test_dist  = vec(sum(y_test,  dims=2))
    println("   Train distribución: $train_dist")
    println("   Test  distribución: $test_dist")
    println("   Balance: $(maximum(train_dist)/minimum(train_dist) < 1.2 ? "✓" : "✗")\n")

    # ----------------------
    # Modelo (2×RNN + Dense) — entrada (T,B,V)
    # ----------------------
    println("🏗️ Construyendo modelo RNN...")
    hidden_1 = 128
    hidden_2 = 128
    model = Sequential([
        RNN(vocab_size, hidden_1; batch_first=false, return_sequences=true,  activation=tanh),
        RNN(hidden_1,   hidden_2; batch_first=false, return_sequences=false, activation=tanh),
        Dense(hidden_2, n_classes)  # logits
    ])

    if CUDA.functional()
        model = model_to_gpu(model)
        println("   🚀 Modelo en GPU")
    end

    params = collect_parameters(model)
    total_params = sum(length(p.data) for p in params)
    println("✅ Modelo construido:")
    println("   Capas: $(length(model.layers))")
    println("   Parámetros totales: $total_params\n")

    # ----------------------
    # Entrenamiento
    # ----------------------
    epochs       = 60
    batch_size   = 128
    opt          = Adam(0.003f0)
    best_val_acc = 0.0f0
    patience_ctr = 0
    patience     = 8

    println("⚙️ Configurando entrenamiento...\n📊 Entrenando modelo...")
    for epoch in 1:epochs
        # TRAIN
        set_training_mode!(model, true)
        tr_batches = create_batches(X_train, y_train, batch_size; shuffle=true)
        epoch_train_losses = Float32[]
        train_correct = 0; train_total = 0

        for (Xb, Yb) in tr_batches
            # (T,B,V)
            OH = to_onehot_TBV(Xb, vocab_size)
            x = Tensor(OH)
            y = Tensor(Yb)
            if CUDA.functional(); x = to_gpu(x); y = to_gpu(y); end

            @assert size(x.data) == (seq_len, size(Yb,2), vocab_size) "x shape inválida"

            logits = forward(model, x)                 # (C,B)
            loss   = crossentropy_from_logits(logits, y)

            zero_grad!(model)
            seed = Tensor(ones(Float32, 1, 1)); if CUDA.functional(); seed = to_gpu(seed); end
            backward(loss, seed)
            step!(opt, params)

            push!(epoch_train_losses, loss.data[1])

            probs = softmax_cols(logits.data)
            predc = [argmax(view(probs, :, i)) for i in 1:size(probs,2)]
            truec = [argmax(view(y.data,   :, i)) for i in 1:size(probs,2)]
            train_correct += sum(predc .== truec)
            train_total   += length(predc)
        end
        train_loss = mean(epoch_train_losses)
        train_acc  = train_correct / train_total

        # VAL
        set_training_mode!(model, false)
        va_batches = create_batches(X_test, y_test, batch_size; shuffle=false)
        epoch_val_losses = Float32[]
        val_correct = 0; val_total = 0

        for (Xb, Yb) in va_batches
            OH = to_onehot_TBV(Xb, vocab_size)
            x = Tensor(OH); y = Tensor(Yb)
            if CUDA.functional(); x = to_gpu(x); y = to_gpu(y); end

            logits = forward(model, x)
            loss   = crossentropy_from_logits(logits, y)
            push!(epoch_val_losses, loss.data[1])

            probs = softmax_cols(logits.data)
            predc = [argmax(view(probs, :, i)) for i in 1:size(probs,2)]
            truec = [argmax(view(y.data,   :, i)) for i in 1:size(probs,2)]
            val_correct += sum(predc .== truec)
            val_total   += length(predc)
        end
        val_loss = mean(epoch_val_losses)
        val_acc  = val_correct / val_total

        println("Epoch $epoch/$epochs - loss: $(round(train_loss,digits=4)) - acc: $(round(100*train_acc,digits=1))% - val_loss: $(round(val_loss,digits=4)) - val_acc: $(round(100*val_acc,digits=1))%")

        if val_acc > best_val_acc
            best_val_acc = val_acc
            patience_ctr = 0
            println("   ↑ Mejor modelo (val_acc: $(round(100*best_val_acc,digits=1))%)")
        else
            patience_ctr += 1
            if patience_ctr ≥ patience
                println("   ⚠️ Early stopping en época $epoch")
                break
            end
        end
        if epoch % 10 == 0
            opt.learning_rate *= 0.5f0
            println("   📉 LR → $(opt.learning_rate)")
        end
    end

    # ----------------------
    # Evaluación + ejemplos
    # ----------------------
    println("\n📈 Resultados finales:")
    println("   Mejor val accuracy: $(round(100*best_val_acc, digits=2))%")

    println("\n🔮 Ejemplos de predicción:")
    set_training_mode!(model, false)
    for _ in 1:3
        idx = rand(1:size(X_test, 2))
        seq = X_test[:, idx:idx]
        OH  = to_onehot_TBV(seq, vocab_size)
        x   = Tensor(OH); if CUDA.functional(); x = to_gpu(x); end
        logits = forward(model, x)
        probs  = vec(softmax_cols(logits.data))
        pred   = argmax(probs)
        truth  = argmax(y_test[:, idx])
        println("   Seq(head): [$(join(seq[1:5,1], ", "))]  | Real: $truth  Pred: $pred  ($(round(100*maximum(probs),digits=1))%)  $(pred==truth ? "✓" : "✗")")
    end

    # Diagnóstico corto
    println("\n📊 Diagnóstico rápido:")
    seq = X_test[:, 1:1]
    OH  = to_onehot_TBV(seq, vocab_size)
    x   = Tensor(OH); if CUDA.functional(); x = to_gpu(x); end
    logits = forward(model, x)
    probs  = softmax_cols(logits.data)
    println("   Output sum (≈1.0): ", sum(probs))
    gflags = [p.grad !== nothing for p in collect_parameters(model) if p.requires_grad]
    println("   Parámetros con grad: $(sum(gflags))/$(length(gflags))")

    println("\n" * "═"^58)
    println("✅ Entrenamiento completado")
end

main()
