# src/Visualizations.jl - Versión FINAL con todas las correcciones

module Visualizations

using Plots
using Printf
using Statistics
using Dates
using ..TensorEngine: Tensor
using ..NeuralNetwork: Sequential, Dense, Conv2D, collect_parameters, relu, sigmoid, tanh_activation, softmax, Activation
using ..ConvolutionalLayers: MaxPooling, Conv2DTranspose
using ..Layers: BatchNorm, Flatten, DropoutLayer, GlobalAvgPool, LayerNorm, RNN, RNNCell
using ..Callbacks: AbstractCallback
 
export plot_training_history, LivePlotter, plot_model_architecture, 
       plot_training_progress, plot_metrics, plot_conv_filters,  moving_average

# Configuración global profesional
function setup_plot_theme()
    gr(size=(800, 400), dpi=100)  # Tamaño por defecto más pequeño
    theme(:default)
end

# Paleta de colores profesional
const DEEP_COLORS = [
    RGB(0.12, 0.47, 0.71),  # Azul profundo
    RGB(1.0, 0.5, 0.05),    # Naranja vibrante
    RGB(0.17, 0.63, 0.17),  # Verde esmeralda
    RGB(0.84, 0.15, 0.16),  # Rojo carmesí
    RGB(0.58, 0.4, 0.74),   # Púrpura
    RGB(0.55, 0.34, 0.29),  # Marrón
]

# Funciones existentes mejoradas
function plot_training_progress(train_losses::Vector{Float64}, val_losses::Union{Nothing, Vector{Float64}}=nothing)
    setup_plot_theme()
    
    p = plot(size=(900, 500),
             background_color=RGB(0.98, 0.98, 0.98),
             background_color_subplot=:white,
             foreground_color=:black,
             margin=15Plots.mm)
    
    epochs = 1:length(train_losses)
    
    plot!(p, epochs, train_losses, 
          label="Training Loss", 
          color=DEEP_COLORS[1],
          linewidth=3,
          marker=:circle,
          markersize=5,
          markeralpha=0.7,
          markerstrokewidth=0,
          grid=true,
          gridstyle=:solid,
          gridalpha=0.15,
          minorgrid=true,
          minorgridalpha=0.08)
          
    if val_losses !== nothing
        plot!(p, epochs, val_losses, 
              label="Validation Loss",
              color=DEEP_COLORS[2],
              linewidth=3,
              marker=:diamond,
              markersize=5,
              markeralpha=0.7,
              markerstrokewidth=0,
              linestyle=:dash)
    end
    
    xlabel!(p, "Epoch", guidefontsize=13, guidefontcolor=RGB(0.3, 0.3, 0.3))
    ylabel!(p, "Loss", guidefontsize=13, guidefontcolor=RGB(0.3, 0.3, 0.3))
    title!(p, "Training Progress", titlefontsize=16, titlefontcolor=RGB(0.2, 0.2, 0.2))
    
    display(p)
end

function plot_metrics(train_losses::Vector{Float64}, val_losses::Vector{Float64})
    plot_training_progress(train_losses, val_losses)
end

function plot_conv_filters(filters::Array{Float64,4})
    setup_plot_theme()
    
    kH, kW, in_channels, out_channels = size(filters)
    num_plots = min(out_channels, 16)
    
    layout = (ceil(Int, sqrt(num_plots)), ceil(Int, sqrt(num_plots)))
    
    p = plot(layout=layout, 
             size=(180*layout[2], 180*layout[1]),
             legend=false,
             margin=5Plots.mm)
    
    for i in 1:num_plots
        filter_img = filters[:, :, 1, i]
        heatmap!(p[i], filter_img, 
                 title="Filter $i",
                 titlefontsize=9,
                 color=:viridis,
                 colorbar=false,
                 aspect_ratio=:equal)
    end
    
    display(p)
end

"""
    plot_training_history(history; figsize=(10,6), save_path=nothing)

Visualiza el historial de entrenamiento con diseño profesional.
"""
function plot_training_history(history::Dict; figsize=(10, 6), save_path=nothing)
    setup_plot_theme()
    
    has_val_loss = haskey(history, :val_loss) &&
               history[:val_loss] !== nothing &&
               !all(isnan.(history[:val_loss]))
    n_metrics = count(k -> k ∉ [:train_loss, :val_loss], keys(history))
    
    layout = n_metrics > 0 ? (1, 2) : (1, 1)
    
    # CAMBIO: Reducir tamaño cuando hay 2 subplots
    actual_size = if layout == (1, 2)
        (figsize[1]*80, figsize[2]*80)  # Más pequeño para 2 subplots
    else
        (figsize[1]*100, figsize[2]*100)
    end
    
    p = plot(layout=layout, 
             size=actual_size,
             margin=15Plots.mm,  # También reducir márgenes
             background_color=RGB(0.98, 0.98, 0.98))
    
    # SUBPLOT 1: PÉRDIDAS con diseño mejorado
    epochs = 1:length(history[:train_loss])
    
    # Fondo sutil
    plot!(p[1], 
          background_color_subplot=:white,
          foreground_color_subplot=RGB(0.2, 0.2, 0.2))
    
    # Línea principal con sombra
    plot!(p[1], epochs, history[:train_loss], 
          label="Training Loss", 
          color=DEEP_COLORS[1],
          linewidth=3.5,
          marker=:circle,
          markersize=5,
          markeralpha=0.8,
          markerstrokewidth=0,
          fillrange=0,
          fillalpha=0.02,
          fillcolor=DEEP_COLORS[1])
    
    if has_val_loss
        plot!(p[1], epochs, history[:val_loss], 
              label="Validation Loss",
              color=DEEP_COLORS[2],
              linewidth=3.5,
              marker=:diamond,
              markersize=5,
              markeralpha=0.8,
              markerstrokewidth=0,
              linestyle=:dashdot)
    end
    
    # Estilo mejorado
    plot!(p[1],
          grid=true,
          gridstyle=:solid,
          gridalpha=0.15,
          gridlinewidth=0.5,
          minorgrid=true,
          minorgridalpha=0.08)
    
    xlabel!(p[1], "Epoch", guidefontsize=13, guidefontcolor=RGB(0.3, 0.3, 0.3))
    ylabel!(p[1], "Loss", guidefontsize=13, guidefontcolor=RGB(0.3, 0.3, 0.3))
    title!(p[1], "Model Loss Evolution", titlefontsize=16, titlefontcolor=RGB(0.2, 0.2, 0.2))
    
    # SUBPLOT 2: MÉTRICAS
    if n_metrics > 0
        plot!(p[2],
              background_color_subplot=:white,
              foreground_color_subplot=RGB(0.2, 0.2, 0.2))
        
        metric_idx = 0
        for (key, values) in history
            if key ∉ [:train_loss, :val_loss]
                metric_idx += 1
                color = DEEP_COLORS[min(metric_idx + 2, length(DEEP_COLORS))]
                
                plot!(p[2], epochs, values,
                      label=replace(string(key), "_" => " ") |> titlecase,
                      color=color,
                      linewidth=3,
                      marker=:auto,
                      markersize=4,
                      markeralpha=0.8,
                      markerstrokewidth=0)
            end
        end
        
        plot!(p[2],
              grid=true,
              gridstyle=:solid,
              gridalpha=0.15,
              gridlinewidth=0.5,
              minorgrid=true,
              minorgridalpha=0.08)
        
        xlabel!(p[2], "Epoch", guidefontsize=13, guidefontcolor=RGB(0.3, 0.3, 0.3))
        ylabel!(p[2], "Metric Value", guidefontsize=13, guidefontcolor=RGB(0.3, 0.3, 0.3))
        title!(p[2], "Training Metrics", titlefontsize=16, titlefontcolor=RGB(0.2, 0.2, 0.2))
    end
    
    if save_path !== nothing
        savefig(p, save_path)
        println("📊 Figura guardada en: $save_path")
    end
    
    display(p)
    return p
end

"""
    LivePlotter <: AbstractCallback

Callback para visualización profesional en tiempo real.
"""
mutable struct LivePlotter <: AbstractCallback
    update_freq::Int              
    metrics::Vector{String}       
    fig                          
    display_handle               
    batch_count::Int             
    loss_history::Vector{Float64}
    metric_history::Dict{String, Vector{Float64}}
    initialized::Bool
    
    epoch_count::Int
    epoch_losses::Vector{Float64}
    epoch_metrics::Dict{String, Vector{Float64}}
    last_update_batch::Int
    
    function LivePlotter(; update_freq::Int=10, metrics::Vector{String}=String[])
        new(update_freq, metrics, nothing, nothing, 0, Float64[], 
            Dict{String, Vector{Float64}}(), false,
            0, Float64[], Dict{String, Vector{Float64}}(), 0)
    end
end

import ..Callbacks: on_train_begin, on_batch_end, on_epoch_end, on_train_end, on_epoch_begin

function on_train_begin(lp::LivePlotter, logs::Dict=Dict())
    setup_plot_theme()
    lp.batch_count = 0
    lp.epoch_count = 0
    lp.loss_history = Float64[]
    lp.metric_history = Dict(m => Float64[] for m in lp.metrics)
    lp.epoch_losses = Float64[]
    lp.epoch_metrics = Dict(m => Float64[] for m in lp.metrics)
    lp.initialized = false
    lp.last_update_batch = 0
end

function on_epoch_begin(lp::LivePlotter, epoch::Int, logs::Dict=Dict())
    lp.epoch_count = epoch
end

function on_batch_end(lp::LivePlotter, batch::Int, logs::Dict=Dict())
    lp.batch_count += 1
    
    if haskey(logs, :loss)
        push!(lp.loss_history, logs[:loss])
    end
    
    for metric in lp.metrics
        metric_sym = Symbol(metric)
        if haskey(logs, metric_sym)
            push!(lp.metric_history[metric], logs[metric_sym])
        end
    end
    
    # Actualizar con frecuencia adaptativa
    should_update = if lp.batch_count < 100
        lp.batch_count % lp.update_freq == 0
    elseif lp.batch_count < 1000
        lp.batch_count % (lp.update_freq * 5) == 0
    else
        lp.batch_count % (lp.update_freq * 20) == 0
    end
    
    if should_update && lp.batch_count > lp.last_update_batch
        lp.last_update_batch = lp.batch_count
        update_live_plot!(lp)
    end
end

function on_epoch_end(lp::LivePlotter, epoch::Int, logs::Dict=Dict())
    lp.epoch_count = epoch
    
    # Calcular promedio de la época
    if !isempty(lp.loss_history)
        recent_losses = lp.loss_history[max(1, end-100):end]
        push!(lp.epoch_losses, mean(recent_losses))
    end
    
    # Guardar métricas
    for metric in lp.metrics
        value = nothing
        for possible_key in [Symbol("val_" * metric), Symbol(metric), 
                           Symbol("train_" * metric), Symbol("val_accuracy")]
            if haskey(logs, possible_key)
                value = logs[possible_key]
                break
            end
        end
        
        if value !== nothing
            if !haskey(lp.epoch_metrics, metric)
                lp.epoch_metrics[metric] = Float64[]
            end
            push!(lp.epoch_metrics[metric], value)
        end
    end
    
    update_live_plot!(lp)
end

function on_train_end(lp::LivePlotter, logs::Dict=Dict())
    update_live_plot!(lp)
    println("\n✅ Entrenamiento completado. Visualización final actualizada.")
end

"""
Actualiza la visualización profesional sin duplicar labels.
"""
function update_live_plot!(lp::LivePlotter)
    if isempty(lp.loss_history) && isempty(lp.epoch_losses)
        return
    end
    
    n_plots = 1 + (length(lp.metrics) > 0 ? 1 : 0)
    
    # CAMBIO: Tamaño adaptativo basado en número de subplots
    plot_height = n_plots == 1 ? 400 : 350
    total_size = (900, plot_height * n_plots)
    
    # SOLUCIÓN CLAVE: Recrear la figura completamente cada vez
    lp.fig = plot(layout=(n_plots, 1), 
                 size=total_size,
                 margin=15Plots.mm,  # Márgenes más pequeños
                 background_color=RGB(0.98, 0.98, 0.98))
    
    # PLOT 1: PÉRDIDA con diseño profesional
    plot!(lp.fig[1],
          background_color_subplot=:white,
          foreground_color_subplot=RGB(0.2, 0.2, 0.2),
          legend=:topright,
          grid=true,
          gridstyle=:solid,
          gridalpha=0.15,
          gridlinewidth=0.5,
          minorgrid=true,
          minorgridalpha=0.08)
    
    # Visualización basada en tipo de datos
    if !isempty(lp.epoch_losses) && length(lp.epoch_losses) >= 2
        # Vista por épocas
        epochs = 1:length(lp.epoch_losses)
        
        # Línea principal con diseño elegante
        plot!(lp.fig[1], 
              epochs, 
              lp.epoch_losses,
              label="Loss (epoch average)",
              color=DEEP_COLORS[1],
              linewidth=4,
              marker=:circle,
              markersize=6,
              markeralpha=0.8,
              markerstrokewidth=0,
              fillrange=0,
              fillalpha=0.03,
              fillcolor=DEEP_COLORS[1])
        
        # Tendencia suavizada
        if length(epochs) >= 3
            alpha = 0.3
            smoothed = similar(lp.epoch_losses)
            smoothed[begin] = lp.epoch_losses[begin]
            for i in axes(smoothed, 1)[begin+1:end]
                smoothed[i] = alpha * lp.epoch_losses[i] + (1 - alpha) * smoothed[i-1]
            end
            
            plot!(lp.fig[1],
                  epochs,
                  smoothed,
                  label="Smoothed trend",
                  color=DEEP_COLORS[3],
                  linewidth=2.5,
                  linestyle=:dash,
                  alpha=0.8)
        end
        
        xlabel!(lp.fig[1], "Epoch", guidefontsize=13, guidefontcolor=RGB(0.3, 0.3, 0.3))
        ylabel!(lp.fig[1], "Loss", guidefontsize=13, guidefontcolor=RGB(0.3, 0.3, 0.3))
        
    else
        # Vista de batches (simplificada)
        total_points = length(lp.loss_history)
        step = max(1, total_points ÷ 200)
        indices = 1:step:total_points
        x_data = collect(indices)
        y_data = [mean(lp.loss_history[max(1, i-step÷2):min(total_points, i+step÷2)]) 
                  for i in indices]
        
        # UNA SOLA línea sin duplicación
        plot!(lp.fig[1],
              x_data,
              y_data,
              label="Training loss",
              color=DEEP_COLORS[1],
              linewidth=3,
              alpha=0.9)
        
        xlabel!(lp.fig[1], "Training Steps", guidefontsize=13, guidefontcolor=RGB(0.3, 0.3, 0.3))
        ylabel!(lp.fig[1], "Loss", guidefontsize=13, guidefontcolor=RGB(0.3, 0.3, 0.3))
    end
    
    # Título informativo con estilo
    title_text = "Training Monitor"
    if lp.epoch_count > 0
        title_text *= " • Epoch $(lp.epoch_count)"
    end
    if !isempty(lp.loss_history)
        current_loss = round(lp.loss_history[end], digits=4)
        title_text *= " • Current Loss: $current_loss"
    end
    
    title!(lp.fig[1], title_text, 
           titlefontsize=16, 
           titlefontcolor=RGB(0.2, 0.2, 0.2))
    
    # PLOT 2: MÉTRICAS con diseño consistente
    if n_plots > 1 && !isempty(lp.metrics)
        plot!(lp.fig[2],
              background_color_subplot=:white,
              foreground_color_subplot=RGB(0.2, 0.2, 0.2),
              legend=:best,
              grid=true,
              gridstyle=:solid,
              gridalpha=0.15,
              gridlinewidth=0.5,
              minorgrid=true,
              minorgridalpha=0.08)
        
        has_data = false
        for (idx, metric) in enumerate(lp.metrics)
            if haskey(lp.epoch_metrics, metric) && !isempty(lp.epoch_metrics[metric])
                has_data = true
                metric_data = lp.epoch_metrics[metric]
                epochs = 1:length(metric_data)
                
                plot!(lp.fig[2],
                      epochs,
                      metric_data,
                      label=titlecase(replace(metric, "_" => " ")),
                      color=DEEP_COLORS[min(idx + 1, length(DEEP_COLORS))],
                      linewidth=3,
                      marker=:circle,
                      markersize=5,
                      markeralpha=0.8,
                      markerstrokewidth=0)
            end
        end
        
        if has_data
            xlabel!(lp.fig[2], "Epoch", guidefontsize=13, guidefontcolor=RGB(0.3, 0.3, 0.3))
            ylabel!(lp.fig[2], "Value", guidefontsize=13, guidefontcolor=RGB(0.3, 0.3, 0.3))
            title!(lp.fig[2], "Training Metrics", titlefontsize=16, titlefontcolor=RGB(0.2, 0.2, 0.2))
        else
            annotate!(lp.fig[2], 0.5, 0.5, 
                     text("Waiting for metric data...", :center, 12, :gray))
        end
    end
    
    display(lp.fig)
end



function moving_average(data::Vector{T}, window::Int) where T
    window = max(1, min(window, length(data)))
    n_results = length(data) - window + 1
    
    if n_results <= 0
        return T[]
    end
    
    result = Vector{T}(undef, n_results)
    current_sum = sum(@view data[1:window])
    result[1] = current_sum / window
    
    @inbounds for i in eachindex(result)[2:end]
        current_sum += data[i+window-1] - data[i-1]
        result[i] = current_sum / window
    end
    
    return result
end

function plot_model_architecture(model; save_path=nothing, show_params=true)
    setup_plot_theme()
    
    layer_info = extract_layer_info(model)
    n_layers = length(layer_info)
    
    fig_height = max(8, n_layers * 1.0)
    fig_width = 12
    
    p = plot(size=(fig_width * 100, fig_height * 100),
             xlims=(0, 10),
             ylims=(0, n_layers + 1),
             aspect_ratio=:equal,
             framestyle=:none,
             legend=false,
             background_color=RGB(0.98, 0.98, 0.98))
    
    y_positions = range(n_layers, 1, length=n_layers)
    
    for (i, (layer, info)) in enumerate(zip(1:n_layers, layer_info))
        y = y_positions[i]
        
        rect_width = 7
        rect_height = 0.8
        x_center = 5
        
        color = get_layer_color(info[:type])
        
        # Sombra sutil
        shadow = Shape([x_center - rect_width/2 + 0.05, x_center + rect_width/2 + 0.05, 
                       x_center + rect_width/2 + 0.05, x_center - rect_width/2 + 0.05],
                      [y - rect_height/2 - 0.05, y - rect_height/2 - 0.05, 
                       y + rect_height/2 - 0.05, y + rect_height/2 - 0.05])
        plot!(p, shadow, fillcolor=RGB(0.8, 0.8, 0.8), linewidth=0, alpha=0.3)
        
        # Capa principal
        rectangle = Shape([x_center - rect_width/2, x_center + rect_width/2, 
                          x_center + rect_width/2, x_center - rect_width/2],
                         [y - rect_height/2, y - rect_height/2, 
                          y + rect_height/2, y + rect_height/2])
        
        plot!(p, rectangle, fillcolor=color, linecolor=:darkgray, linewidth=1.5)
        
        # Texto mejorado
        layer_text = info[:name]
        if show_params && info[:params] > 0
            layer_text *= "\n$(format_params(info[:params])) params"
        end
        
        annotate!(p, x_center, y, text(layer_text, :center, 11, :black, :bold))
        
        # Shape info
        if info[:output_shape] !== nothing
            shape_text = string(info[:output_shape])
            annotate!(p, x_center + rect_width/2 + 0.7, y, 
                     text(shape_text, :left, 9, RGB(0.4, 0.4, 0.4)))
        end
        
        # Conexión elegante
        if i < n_layers
            y_next = y_positions[i + 1]
            xs = [x_center, x_center]
            ys = [y - rect_height/2 - 0.1, y_next + rect_height/2 + 0.1]
            plot!(p, xs, ys;              # dos vectores, no cuatro scalars
                linecolor=RGB(0.4,0.4,0.4),
                linewidth=2)
        end
    end
    
    title!(p, "Model Architecture", 
           titlefontsize=18, 
           titlefontcolor=RGB(0.2, 0.2, 0.2))
    
    if save_path !== nothing
        savefig(p, save_path)
        println("🏗️ Arquitectura guardada en: $save_path")
    end
    
    display(p)
    return p
end

# Sobrecarga de plot_training_history para aceptar directamente el objeto History
function plot_training_history(h; kwargs...)
    # Construir un Dict con los datos que espera la versión original
    history = Dict{Symbol, Any}()
    history[:train_loss] = h.train_loss
    history[:val_loss]   = hasproperty(h, :val_loss) ? h.val_loss : nothing
    if hasproperty(h, :train_metrics)
        for (m, vals) in h.train_metrics
            history[Symbol("train_" * m)] = vals
        end
    end
    if hasproperty(h, :val_metrics)
        for (m, vals) in h.val_metrics
            history[Symbol("val_" * m)] = vals
        end
    end
    # Llamar a la versión que procesa Dict
    return plot_training_history(history; kwargs...)
end


function extract_layer_info(model)
    layer_info = []
    
    if !isa(model, Sequential)
        return layer_info
    end
    
    for (i, layer) in enumerate(model.layers)
        info = Dict{Symbol, Any}()
        layer_type = typeof(layer)
        
        if layer_type <: Dense
            info[:type] = "Dense"
            in_features = size(layer.weights.data, 2)
            out_features = size(layer.weights.data, 1)
            info[:name] = "Dense($(in_features)→$(out_features))"
            info[:params] = length(layer.weights.data) + length(layer.biases.data)
            info[:output_shape] = (out_features,)
        elseif layer_type <: RNNCell
            info[:type] = "RNNCell"
            in_features  = layer.input_size
            hidden       = layer.hidden_size
            has_bias     = (layer.b_ih !== nothing)  # si hay b_ih asumimos b_hh también
            info[:name]  = "RNNCell($(in_features)→$(hidden))"
            info[:params] = in_features * hidden + hidden * hidden + (has_bias ? 2 * hidden : 0)
            info[:output_shape] = nothing

        elseif layer_type <: RNN
            info[:type] = "RNN"
            in_features  = layer.cell.input_size
            hidden       = layer.cell.hidden_size
            has_bias     = (layer.cell.b_ih !== nothing)
            seqflag      = layer.return_sequences ? "seq" : "last"
            info[:name]  = "RNN($(in_features)→$(hidden), $seqflag)"
            info[:params] = in_features * hidden + hidden * hidden + (has_bias ? 2 * hidden : 0)
            info[:output_shape] = nothing

        elseif layer_type <: Conv2D
            info[:type] = "Conv2D"
            out_channels = size(layer.weights.data, 1)
            in_channels = size(layer.weights.data, 2)
            kH = size(layer.weights.data, 3)
            kW = size(layer.weights.data, 4)
            kernel_str = "$(kH)×$(kW)"
            info[:name] = "Conv2D($(in_channels)→$(out_channels), $kernel_str)"
            info[:params] = length(layer.weights.data) + length(layer.bias.data)
            info[:output_shape] = nothing
        elseif layer_type <: LayerNorm
            info[:type] = "LayerNorm"
            shape_str = join(layer.normalized_shape, "×")
            info[:name] = "LayerNorm($shape_str)"
            info[:params] = length(layer.gamma.data) + length(layer.beta.data)
            info[:output_shape] = layer.normalized_shape
            
        elseif layer_type <: MaxPooling
            info[:type] = "MaxPool"
            pool_str = layer.pool_size isa Tuple ? 
                "$(layer.pool_size[1])×$(layer.pool_size[2])" : 
                "$(layer.pool_size)×$(layer.pool_size)"
            info[:name] = "MaxPool($pool_str)"
            info[:params] = 0
            info[:output_shape] = nothing
            
        elseif layer_type <: BatchNorm
            info[:type] = "BatchNorm"
            num_features = length(layer.gamma.data)
            info[:name] = "BatchNorm($num_features)"
            info[:params] = 2 * num_features
            info[:output_shape] = nothing
            
        elseif layer_type <: Flatten
            info[:type] = "Flatten"
            info[:name] = "Flatten()"
            info[:params] = 0
            info[:output_shape] = nothing
            
        elseif layer_type <: DropoutLayer
            info[:type] = "Dropout"
            info[:name] = "Dropout($(layer.rate))"
            info[:params] = 0
            info[:output_shape] = nothing
            
        elseif layer_type <: GlobalAvgPool
            info[:type] = "GlobalAvgPool"
            info[:name] = "GlobalAvgPool"
            info[:params] = 0
            info[:output_shape] = nothing
            
        elseif layer_type <: Activation
            info[:type] = "Activation"
            info[:params] = 0
            info[:output_shape] = nothing
            info[:name] = identify_activation(layer)
            
        else
            if applicable(layer, Tensor(Float32[1.0]))
                info[:type] = "Activation"
                info[:params] = 0
                info[:output_shape] = nothing
                info[:name] = identify_activation(layer)
            else
                info[:type] = "Unknown"
                info[:name] = "Unknown Layer"
                info[:params] = 0
                info[:output_shape] = nothing
            end
        end
        
        push!(layer_info, info)
    end
    
    return layer_info
end

function identify_activation(layer)
    try
        test_input = Tensor(Float32[-2.0, -1.0, 0.0, 1.0, 2.0])
        output = layer(test_input)
        result = output.data
        
        if result[1] ≈ 0.0f0 && result[2] ≈ 0.0f0 && result[3] ≈ 0.0f0 && 
           result[4] ≈ 1.0f0 && result[5] ≈ 2.0f0
            return "ReLU"
        elseif all(0 .< result .< 1) && 0.45f0 < result[3] < 0.55f0
            return "Sigmoid"
        elseif all(-1 .< result .< 1) && abs(result[3]) < 0.01f0
            return "Tanh"
        elseif all(result .> 0) && 0.99f0 < sum(result) < 1.01f0
            return "Softmax"
        elseif result[1] < 0 && result[1] > -0.5f0 && result[5] ≈ 2.0f0
            return "LeakyReLU"
        else
            return "Custom Activation"
        end
    catch e
        func_str = string(layer)
        if occursin("relu", lowercase(func_str)) || occursin("#1#", func_str)
            return "ReLU"
        elseif occursin("sigmoid", lowercase(func_str))
            return "Sigmoid"
        elseif occursin("tanh", lowercase(func_str))
            return "Tanh"
        elseif occursin("softmax", lowercase(func_str))
            return "Softmax"
        else
            return "Unknown Activation"
        end
    end
end

function get_layer_color(layer_type::String)
    color_map = Dict(
        "Dense" => :lightblue,
        "Conv2D" => :lightgreen,
        "MaxPool" => :lightyellow,
        "BatchNorm" => :lightcoral,
        "Dropout" => :lightgray,
        "Activation" => :lightpink,
        "Flatten" => :wheat,
        "GlobalAvgPool" => :palegreen,
        "Unknown" => :white,
        "RNN"      => :lightsteelblue,
        "RNNCell"  => :thistle,
    )
    
    return get(color_map, layer_type, :white)
end

function format_params(n::Int)
    if n >= 1_000_000
        return @sprintf("%.1fM", n / 1_000_000)
    elseif n >= 1_000
        return @sprintf("%.1fK", n / 1_000)
    else
        return string(n)
    end
end

end  # module Visualizations