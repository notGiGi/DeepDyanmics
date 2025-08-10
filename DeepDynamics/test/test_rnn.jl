# test/test_rnn.jl
using Test
using Random
using Statistics
using CUDA
using DeepDynamics
using DeepDynamics.Losses: categorical_crossentropy
using DeepDynamics.Optimizers: Adam, step!
using DeepDynamics.TensorEngine
Random.seed!(1234)

# ---------- Utils ----------
# One-hot en layout T,N,V (tiempo, batch, features)
function to_onehot_TNB(X::AbstractMatrix{<:Integer}, V::Int)
    T, N = size(X)
    OH = zeros(Float32, T, N, V)
    @inbounds for n in 1:N, t in 1:T
        OH[t, n, X[t,n]] = 1f0
    end
    OH
end

# Softmax por columnas (para comprobaciones numéricas)
softmax_cols(A) = begin
    m = maximum(A; dims=1)
    ex = exp.(A .- m)
    ex ./ sum(ex; dims=1)
end

# Construye cabeza con 2×RNN + Dense (+ opcional softmax), salida 2D (C,N)
function build_rnn_classifier(in_dim, h1, h2, out_dim; batch_first=false, activation=tanh, with_softmax=true)
    layers = Any[
        RNN(in_dim, h1; batch_first=batch_first, return_sequences=true,  activation=activation),
        RNN(h1,   h2;  batch_first=batch_first, return_sequences=false, activation=activation),
        Dense(h2, out_dim)
    ]
    if with_softmax
        push!(layers, Activation(softmax))
    end
    Sequential(layers)
end

# Construye solo pila recurrente con salida secuencial 3D
function build_rnn_sequence(in_dim, h1, h2; batch_first=false, activation=tanh)
    Sequential([
        RNN(in_dim, h1; batch_first=batch_first, return_sequences=true, activation=activation),
        RNN(h1,   h2;  batch_first=batch_first, return_sequences=true, activation=activation)
    ])
end

to_gpu_if(x) = CUDA.functional() ? to_gpu(x) : x
model_to_gpu_if(m) = CUDA.functional() ? model_to_gpu(m) : m
back_seed() = Tensor(ones(Float32,1,1)) |> to_gpu_if

# ---------- TESTS ----------
@testset "RNN & Embedding suite" begin

    @testset "RNN: forward shapes (CPU)" begin
        T, N, D, H1, H2, C = 4, 3, 5, 6, 7, 4
        X = rand(1:D, T, N)
        OH = to_onehot_TNB(X, D)
        x  = Tensor(OH)

        # Caso 1: salida secuencial 3D (T,N,H2)
        m_seq = build_rnn_sequence(D, H1, H2; batch_first=false)
        y_seq = forward(m_seq, x)
        @test size(y_seq.data) == (T, N, H2)

        # Caso 2: clasificador (C,N)
        m_clf = build_rnn_classifier(D, H1, H2, C; batch_first=false)
        y_clf = forward(m_clf, x)
        @test size(y_clf.data) == (C, N)

        # batch_first=true, salida secuencial 3D (N,T,H2)
        xbf = Tensor(permutedims(OH, (2,1,3)))  # (N,T,D)
        m_seq_bf = build_rnn_sequence(D, H1, H2; batch_first=true)
        y_seq_bf = forward(m_seq_bf, xbf)
        @test size(y_seq_bf.data) == (N, T, H2)

        # batch_first=true, clasificador (C,N)
        m_clf_bf = build_rnn_classifier(D, H1, H2, C; batch_first=true)
        y_clf_bf = forward(m_clf_bf, xbf)
        @test size(y_clf_bf.data) == (C, N)
    end

    @testset "RNN: CPU/GPU parity (forward)" begin
        T, N, D, H1, H2, C = 3, 2, 5, 6, 7, 4
        X = rand(1:D, T, N)
        OH = to_onehot_TNB(X, D)
        x_cpu = Tensor(OH)
        m_cpu = build_rnn_classifier(D, H1, H2, C; batch_first=false, with_softmax=false)
        y_cpu = forward(m_cpu, x_cpu)               # (C,N)

        if CUDA.functional()
            m_gpu = model_to_gpu(m_cpu)
            x_gpu = to_gpu(Tensor(OH))
            y_gpu = forward(m_gpu, x_gpu)
            @test size(y_gpu.data) == size(y_cpu.data)
            @test maximum(abs.(Array(y_gpu.data) .- y_cpu.data)) ≤ 1e-4
        else
            @test true
        end
    end

    @testset "RNN: backward + CCE reduce loss" begin
        T, N, D, H1, H2, C = 5, 8, 6, 16, 16, 3
        X = rand(1:D, T, N)
        Y = zeros(Float32, C, N); for n in 1:N; Y[rand(1:C), n] = 1; end
        OH = to_onehot_TNB(X, D)
        x  = Tensor(OH) |> to_gpu_if
        y  = Tensor(Y)  |> to_gpu_if

        model = build_rnn_classifier(D, H1, H2, C; batch_first=false, with_softmax=true) |> model_to_gpu_if
        opt = Adam(0.01f0)

        set_training_mode!(model, true)
        ŷ1 = forward(model, x)
        ℓ1 = categorical_crossentropy(ŷ1, y)
        zero_grad!(model); backward(ℓ1, back_seed()); step!(opt, collect_parameters(model))
        ŷ2 = forward(model, x)
        ℓ2 = categorical_crossentropy(ŷ2, y)

        @test ℓ2.data[1] ≤ ℓ1.data[1] + 1e-6
    end

    @testset "RNN: gradiente hacia h0 explícito" begin
        T, N, D, H, C = 4, 3, 5, 8, 3
        X = rand(1:D, T, N); OH = to_onehot_TNB(X, D)
        x  = Tensor(OH) |> to_gpu_if

        # Armamos las capas manualmente
        rnn1 = RNN(D, H; batch_first=false, return_sequences=true,  activation=tanh)
        rnn2 = RNN(H, H; batch_first=false, return_sequences=false, activation=tanh)
        fc   = Dense(H, C)

        # Empaquetar y mover (si hay GPU), luego re-vincular referencias
        dev_model = model_to_gpu_if(Sequential([rnn1, rnn2, fc]))
        rnn1 = dev_model.layers[1]
        rnn2 = dev_model.layers[2]
        fc   = dev_model.layers[3]

        # h0 explícito con gradiente
        h0 = Tensor(zeros(Float32, H, N); requires_grad=true) |> to_gpu_if

        # forward manual respetando h0 SOLO en rnn1 (usar forward, no call)
        y1 = forward(rnn1, x, h0)      # (T,N,H)
        y2 = forward(rnn2, y1)         # (H,N)
        y3 = fc(y2)                    # (C,N)
        ŷ  = Activation(softmax)(y3)  # callable

        # Target dummy
        Y = zeros(Float32, C, N); for n in 1:N; Y[rand(1:C),n] = 1; end
        y = Tensor(Y) |> to_gpu_if

        ℓ = categorical_crossentropy(ŷ, y)
        zero_grad!(dev_model); backward(ℓ, back_seed())
        @test h0.grad !== nothing
    end

    @testset "Embedding: forward shapes y padding=0" begin
        V, E = 7, 6
        T, N = 5, 4
        # índices con ceros (padding)
        X = zeros(Int, T, N)
        for n in 1:N, t in 1:T
            X[t,n] = rand() < 0.2 ? 0 : rand(1:V)
        end
        emb = Embedding(V, E)
        out = forward(emb, Tensor(X))  # se espera (T,N,E) por tu implementación

        sz = size(out.data)
        @test sz == (T, N, E)

        data = out.data isa CUDA.CuArray ? Array(out.data) : out.data
        for t in 1:T, n in 1:N
            if X[t,n] == 0
                v = @view data[t,n,:]
                @test sqrt(sum(abs2, v)) ≤ 1e-6
            end
        end
    end

    @testset "Embedding: backward acumula solo en índices usados (CPU/GPU)" begin
        V, E = 8, 5
        T, N = 4, 3
        X = [1 0 2;
             3 4 0;
             1 2 5;
             0 4 2]
        @test size(X) == (T, N)

        # CPU
        emb = Embedding(V, E)
        out = forward(emb, Tensor(X))
        TensorEngine.backward(out, Tensor(ones(Float32, size(out.data))))
        @test emb.weights.grad !== nothing

        used = unique(vec(X[X .> 0]))
        g = emb.weights.grad.data
        colnorms = [sqrt(sum(abs2, @view g[:,i])) for i in 1:V]
        for i in 1:V
            if i in used
                @test colnorms[i] > 0
            else
                @test colnorms[i] ≤ 1e-8
            end
        end

        # GPU (si disponible)
        if CUDA.functional()
            emb_g = Embedding(V, E)
            emb_g.weights = to_gpu(emb_g.weights)
            outg = forward(emb_g, Tensor(X))
            TensorEngine.backward(outg, to_gpu(Tensor(ones(Float32, size(outg.data)))))
            @test emb_g.weights.grad !== nothing
        else
            @test true
        end
    end

    @testset "RNN: sanity mini-overfit (rápido)" begin
        T, N, D, H1, H2, C = 6, 32, 10, 32, 32, 3
        # dataset sintético con patrón por clase
        X = zeros(Int, T, N)
        Y = zeros(Float32, C, N)
        for n in 1:N
            cls = rand(1:C)
            Y[cls,n] = 1
            for t in 1:T
                X[t,n] = cls == 1 ? rand(1:3) : cls == 2 ? rand(4:7) : rand(8:10)
            end
        end
        OH = to_onehot_TNB(X, D)
        x = Tensor(OH) |> to_gpu_if
        y = Tensor(Y)  |> to_gpu_if

        model = build_rnn_classifier(D, H1, H2, C; batch_first=false, with_softmax=true) |> model_to_gpu_if
        opt = Adam(0.01f0)

        set_training_mode!(model, true)
        acc = 0.0
        for step in 1:80
            ŷ = forward(model, x)
            ℓ = categorical_crossentropy(ŷ, y)
            zero_grad!(model)
            backward(ℓ, back_seed())
            step!(opt, collect_parameters(model))

            p = ŷ.data
            preds = [argmax(@view p[:,i]) for i in 1:N]
            truth = [argmax(@view y.data[:,i]) for i in 1:N]
            acc = sum(preds .== truth) / N
        end
        @test acc ≥ 0.95
    end
end
