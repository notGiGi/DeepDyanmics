# demo_advanced.jl
using Pkg
Pkg.activate(".")                       # Activa tu entorno DeepDynamics
using DeepDynamics                       # Trae fit!, Adam, categorical_crossentropy, etc.
using DeepDynamics.Visualizations        # Trae todas las gráficas y callbacks
using DeepDynamics.TensorEngine: Tensor
using DeepDynamics.NeuralNetwork: Sequential, Dense, Activation, relu, softmax
using DeepDynamics.Layers: BatchNorm, DropoutLayer
using DeepDynamics.Callbacks: EarlyStopping, ReduceLROnPlateau
using Random, Statistics

# 1) Generar dataset sintético con 4 clústeres en ℝ³
function generate_4_clusters(n_per_class=250)
    Random.seed!(2025)
    centers = [
        Float32[ 2.0,  2.0,  2.0],
        Float32[-2.0, -2.0,  2.0],
        Float32[ 2.0, -2.0, -2.0],
        Float32[-2.0,  2.0, -2.0],
    ]
    X, y = Tensor[], Tensor[]
    for (c, center) in enumerate(centers)
        for _ in 1:n_per_class
            push!(X, Tensor(randn(Float32,3,1) .+ center))
            label = zeros(Float32, 4, 1)
            label[c,1] = 1f0
            push!(y, Tensor(label))
        end
    end
    perm = shuffle(1:length(X))
    return X[perm], y[perm]
end

# 2) Preparar datos y modelo profundo
X, y = generate_4_clusters()   # 4×250 = 1000 muestras

model = Sequential([
    Dense(3, 128),         BatchNorm(128), Activation(relu), DropoutLayer(0.4),
    Dense(128, 64),        Activation(relu), BatchNorm(64),
    Dense(64, 32),         Activation(relu), DropoutLayer(0.3),
    Dense(32, 16),         Activation(relu),
    Dense(16, 4),          Activation(softmax)
])

# 3) Callbacks avanzados
liveplot  = LivePlotter(update_freq=20, metrics=["accuracy"])
reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5f0, patience=3, min_lr=1e-6f0)
earlystop = EarlyStopping(patience=5, min_delta=0.0005f0)

# 4) Entrenar
history = fit!(model, X, y;
               epochs=100,
               batch_size=64,
               validation_split=0.2f0,
               optimizer=Adam(0.005f0),
               loss_fn=categorical_crossentropy,
               metrics=[:accuracy],
               callbacks=[liveplot, reduce_lr, earlystop],
               verbose=true)

# 5) Graficar historial completo
plot_training_history(history; save_path="advanced_training_history.png")

# 6) Graficar media móvil de la pérdida (ventana=5)
plot_moving_average(history.train_loss, 5;
                    title="Smoothed Training Loss (MA=5)",
                    xlabel="Epoch", ylabel="Loss")

# 7) Diagrama de arquitectura
plot_model_architecture(model; show_params=true, save_path="advanced_architecture.png")

# 8) Ejemplo de filtros convolucionales aleatorios
filters = randn(Float64, 5,5,1,12)   # 12 filtros 5×5 en 1 canal
plot_conv_filters(filters; cols=4)

println("✅ Demo avanzado completado — revisa los PNGs y las gráficas en pantalla.")
