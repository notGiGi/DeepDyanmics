module ReportGenerator

using Dates
using Mustache
using Plots
using PlotlyJS
using PrettyTables
using Markdown
using LaTeXStrings
using JSON
using Statistics
using CUDA

using ..TensorEngine
using ..NeuralNetwork
using ..Training: History
using ..Callbacks: AbstractCallback
using ..Visualizations: setup_plot_theme, get_layer_color, format_params

# Importar explícitamente las funciones que vamos a extender
import ..Callbacks: on_train_begin, on_train_end, on_epoch_end

export generate_training_report, ReportTemplate, ReportCallback, 
       create_default_template, export_to_pdf, export_to_latex



# =============================================================================
# Report Templates
# =============================================================================
struct ReportTemplate
    header::String
    css_style::String
    plot_config::Dict{String,Any}
    sections::Vector{Symbol}
    
    function ReportTemplate(;
        header::String = "DeepDynamics Training Report",
        css_style::String = default_css_style(),
        plot_config::Dict = default_plot_config(),
        sections::Vector{Symbol} = default_sections()
    )
        new(header, css_style, plot_config, sections)
    end
end

function default_sections()
    return [:executive_summary, :model_architecture, :training_progress,
            :hyperparameters, :performance_analysis, :hardware_utilization,
            :reproducibility_info]
end

function default_css_style()
    return """
    <style>
        body { 
            font-family: 'Segoe UI', Arial, sans-serif; 
            line-height: 1.6; 
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .report-container {
            background: white;
            padding: 40px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        h1 { 
            color: #2c3e50; 
            border-bottom: 3px solid #3498db; 
            padding-bottom: 10px;
            margin-bottom: 30px;
        }
        h2 { 
            color: #34495e; 
            margin-top: 30px;
            margin-bottom: 20px;
        }
        .metric-box {
            background: #ecf0f1;
            padding: 15px;
            border-radius: 5px;
            margin: 10px 0;
            display: inline-block;
            min-width: 200px;
        }
        .metric-value {
            font-size: 24px;
            font-weight: bold;
            color: #3498db;
        }
        .metric-label {
            font-size: 14px;
            color: #7f8c8d;
        }
        table {
            border-collapse: collapse;
            width: 100%;
            margin: 20px 0;
        }
        th, td {
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }
        th {
            background-color: #3498db;
            color: white;
        }
        tr:nth-child(even) {
            background-color: #f2f2f2;
        }
        .code-block {
            background: #2c3e50;
            color: #ecf0f1;
            padding: 20px;
            border-radius: 5px;
            overflow-x: auto;
            font-family: 'Courier New', monospace;
            margin: 20px 0;
        }
        .warning {
            background: #f39c12;
            color: white;
            padding: 10px;
            border-radius: 5px;
            margin: 10px 0;
        }
        .success {
            background: #27ae60;
            color: white;
            padding: 10px;
            border-radius: 5px;
            margin: 10px 0;
        }
        .plot-container {
            margin: 20px 0;
            text-align: center;
        }
    </style>
    """
end

function default_plot_config()
    return Dict{String,Any}(
        "theme" => :seaborn,
        "size" => (800, 500),
        "dpi" => 100,
        "background_color" => :white,
        "foreground_color" => :black
    )
end

# =============================================================================
# Report Generation Functions
# =============================================================================

"""
    generate_training_report(model, history, config; format=:html, template=ReportTemplate())
    
Genera un reporte completo del entrenamiento del modelo.
"""
function generate_training_report(
    model,
    history::History,
    config::Dict{String,T};
    format::Symbol = :html,
    template::ReportTemplate = ReportTemplate(),
    save_path::String = "training_report"
) where {T}

    # ————————————————————————————————————————————————————————————
    # Aceptar cualquier Dict{String,T} para config, y trabajar con un
    # Dict{String,Any} internamente para que las asignaciones funcionen
    # ————————————————————————————————————————————————————————————
    cfg = Dict{String,Any}(config)

    # Validar formato
    @assert format in (:html, :pdf, :markdown, :latex) "Formato no soportado: $format"

    # Recopilar toda la información
    report_data = collect_report_data(model, history, cfg)

    # Generar el contenido del reporte
    if format == :html
        report_content = generate_html_report(report_data, template)
        filename = save_path * ".html"
    elseif format == :pdf
        report_content = generate_html_report(report_data, template)
        filename = export_to_pdf(report_content, save_path)
    elseif format == :markdown
        report_content = generate_markdown_report(report_data, template)
        filename = save_path * ".md"
    elseif format == :latex
        report_content = generate_latex_report(report_data, template)
        filename = save_path * ".tex"
    end

    # Guardar a disco
    open(filename, "w") do io
        write(io, report_content)
    end

    println("📊 Reporte generado: $filename")
    return filename
end


# =============================================================================
# Data Collection
# =============================================================================

function collect_report_data(model, history::History, config::Dict{String,Any})
    data = Dict{String,Any}()
    
    # Metadatos básicos
    data["timestamp"] = Dates.now()
    data["total_epochs"] = history.epochs
    data["config"] = config
    
    # Executive Summary
    data["executive_summary"] = generate_executive_summary(history, config)
    
    # Model Architecture
    data["model_architecture"] = extract_model_architecture(model)
    
    # Training Progress
    data["training_progress"] = extract_training_progress(history)
    
    # Performance Analysis
    data["performance_analysis"] = analyze_performance(history)
    
    # Hardware Utilization
    data["hardware_utilization"] = collect_hardware_info()
    
    # Reproducibility
    data["reproducibility"] = collect_reproducibility_info(config)
    
    return data
end

function generate_executive_summary(history::History, config::Dict{String,Any})
    summary = Dict{String,Any}()
    
    # Mejores métricas
    summary["best_train_loss"] = isempty(history.train_loss) ? NaN : minimum(history.train_loss)
    summary["best_val_loss"] = isempty(history.val_loss) ? NaN : minimum(filter(!isnan, history.val_loss))
    summary["final_train_loss"] = isempty(history.train_loss) ? NaN : history.train_loss[end]
    summary["final_val_loss"] = isempty(history.val_loss) ? NaN : history.val_loss[end]
    
    # Tiempo total (estimado)
    total_time = get(config, "training_time", 0.0)
    summary["total_time"] = total_time
    summary["avg_time_per_epoch"] = total_time / max(1, history.epochs)
    
    # Métricas adicionales
    for (metric_name, values) in history.train_metrics
        if !isempty(values)
            summary["best_train_$metric_name"] = maximum(values)
            summary["final_train_$metric_name"] = values[end]
        end
    end
    
    for (metric_name, values) in history.val_metrics
        clean_values = filter(!isnan, values)
        if !isempty(clean_values)
            summary["best_val_$metric_name"] = maximum(clean_values)
            summary["final_val_$metric_name"] = clean_values[end]
        end
    end
    
    # Determinar si hubo overfitting
    if !isempty(history.val_loss) && !isempty(history.train_loss) && length(history.val_loss) > 10
        train_end = length(history.train_loss)
        train_start = max(1, train_end - 5)
        val_end = length(history.val_loss)
        val_start = max(1, val_end - 5)
        
        if train_start < train_end && val_start < val_end
            train_trend = history.train_loss[train_end] - history.train_loss[train_start]
            val_trend = history.val_loss[val_end] - history.val_loss[val_start]
            summary["potential_overfitting"] = train_trend < 0 && val_trend > 0
        else
            summary["potential_overfitting"] = false
        end
    else
        summary["potential_overfitting"] = false
    end
    
    return summary
end

function extract_model_architecture(model)
    arch_info = Dict{String,Any}()
    
    # Información básica
    arch_info["type"] = string(typeof(model))
    
    # Contar parámetros
    total_params = 0
    trainable_params = 0
    
    params = NeuralNetwork.collect_parameters(model)
    for p in params
        param_count = length(p.data)
        total_params += param_count
        trainable_params += param_count  # Por ahora todos son entrenables
    end
    
    arch_info["total_parameters"] = total_params
    arch_info["trainable_parameters"] = trainable_params
    
    # Extraer información de capas
    if isa(model, NeuralNetwork.Sequential)
        layers_info = []
        for (i, layer) in enumerate(model.layers)
            layer_data = extract_layer_info(layer, i)
            push!(layers_info, layer_data)
        end
        arch_info["layers"] = layers_info
    end
    
    return arch_info
end

function extract_layer_info(layer, index::Int)
    info = Dict{String,Any}()
    info["index"] = index
    info["type"] = string(typeof(layer))
    
    # Información específica por tipo de capa
    if isa(layer, NeuralNetwork.Dense)
        info["name"] = "Dense"
        info["input_size"] = size(layer.weights.data, 2)
        info["output_size"] = size(layer.weights.data, 1)
        info["parameters"] = length(layer.weights.data) + length(layer.biases.data)
        info["activation"] = "linear"
    elseif isa(layer, NeuralNetwork.Activation)
        info["name"] = "Activation"
        info["parameters"] = 0
        # Intentar identificar la función de activación
        info["function"] = "custom"
    elseif isa(layer, Function)
        info["name"] = "Lambda"
        info["parameters"] = 0
        info["function"] = string(layer)
    else
        info["name"] = "Custom"
        info["parameters"] = 0
    end
    
    return info
end

function extract_training_progress(history::History)
    progress = Dict{String,Any}()
    
    # Pérdidas
    progress["train_loss"] = history.train_loss
    progress["val_loss"] = history.val_loss
    
    # Métricas
    progress["train_metrics"] = history.train_metrics
    progress["val_metrics"] = history.val_metrics
    
    # Calcular estadísticas
    if !isempty(history.train_loss) && length(history.train_loss) > 1
        progress["loss_reduction"] = (history.train_loss[1] - history.train_loss[end]) / history.train_loss[1] * 100
    else
        progress["loss_reduction"] = 0.0
    end
    
    return progress
end

function analyze_performance(history::History)
    analysis = Dict{String,Any}()
    
    # Análisis por época
    epochs_data = []
    for epoch in 1:history.epochs
        epoch_info = Dict{String,Any}()
        epoch_info["epoch"] = epoch
        
        if epoch <= length(history.train_loss)
            epoch_info["train_loss"] = history.train_loss[epoch]
        end
        
        if epoch <= length(history.val_loss) && !isempty(history.val_loss)
            epoch_info["val_loss"] = history.val_loss[epoch]
        end
        
        # Agregar métricas
        for (metric_name, values) in history.train_metrics
            if epoch <= length(values)
                epoch_info["train_$metric_name"] = values[epoch]
            end
        end
        
        for (metric_name, values) in history.val_metrics
            if epoch <= length(values)
                epoch_info["val_$metric_name"] = values[epoch]
            end
        end
        
        push!(epochs_data, epoch_info)
    end
    
    analysis["epochs"] = epochs_data
    
    # Identificar mejor época
    if !isempty(history.val_loss)
        clean_val_losses = [(i, v) for (i, v) in enumerate(history.val_loss) if !isnan(v)]
        if !isempty(clean_val_losses)
            best_epoch_idx = argmin([v for (i, v) in clean_val_losses])
            analysis["best_epoch"] = clean_val_losses[best_epoch_idx][1]
        else
            analysis["best_epoch"] = 1
        end
    elseif !isempty(history.train_loss)
        analysis["best_epoch"] = argmin(history.train_loss)
    else
        analysis["best_epoch"] = 1
    end
    
    return analysis
end

function collect_hardware_info()
    hardware = Dict{String,Any}()
    
    # Información de CPU
    hardware["cpu_threads"] = Threads.nthreads()
    
    # Información de GPU
    if CUDA.functional()
        hardware["gpu_available"] = true
        hardware["gpu_device"] = CUDA.device()
        hardware["gpu_name"] = CUDA.name(CUDA.device())
        
        # Memoria GPU - CUDA.memory_info() devuelve una tupla (free, total)
        mem_info = CUDA.memory_info()
        if isa(mem_info, Tuple) && length(mem_info) >= 2
            free_bytes = mem_info[1]
            total_bytes = mem_info[2]
        else
            # Fallback si la API cambió
            free_bytes = CUDA.available_memory()
            total_bytes = CUDA.total_memory()
        end
        
        hardware["gpu_memory_total"] = total_bytes / 1024^3  # GB
        hardware["gpu_memory_free"] = free_bytes / 1024^3    # GB
        hardware["gpu_memory_used"] = (total_bytes - free_bytes) / 1024^3  # GB
    else
        hardware["gpu_available"] = false
    end
    
    return hardware
end

function collect_reproducibility_info(config::Dict{String,Any})
    repro = Dict{String,Any}()
    
    # Semilla aleatoria
    repro["random_seed"] = get(config, "random_seed", "not set")
    
    # Hiperparámetros
    repro["batch_size"] = get(config, "batch_size", "unknown")
    repro["learning_rate"] = get(config, "learning_rate", "unknown")
    repro["optimizer"] = get(config, "optimizer", "unknown")
    repro["epochs"] = get(config, "epochs", "unknown")
    
    # Versión de Julia y paquetes
    repro["julia_version"] = string(VERSION)
    
    # Timestamp
    repro["training_date"] = get(config, "start_time", now())
    
    return repro
end
# =============================================================================
# Utility Functions
# =============================================================================


function format_number(n::Number)
    if n >= 1e9
        return "$(round(n/1e9, digits=2))B"
    elseif n >= 1e6
        return "$(round(n/1e6, digits=2))M"
    elseif n >= 1e3
        return "$(round(n/1e3, digits=2))K"
    else
        return string(Int(n))
    end
end

function format_config_value(value)
    if isa(value, Number)
        return string(value)
    elseif isa(value, String)
        return value
    elseif isa(value, Symbol)
        return string(value)
    elseif isa(value, Array)
        return "[" * join(value, ", ") * "]"
    else
        return string(value)
    end
end

function export_to_pdf(html_content::String, output_path::String)
    # Guardar HTML temporal
    temp_html = tempname() * ".html"
    open(temp_html, "w") do io
        write(io, html_content)
    end
    
    # Convertir a PDF usando herramienta externa (requiere wkhtmltopdf o similar)
    pdf_path = output_path * ".pdf"
    
    try
        # Intentar con wkhtmltopdf si está disponible
        run(`wkhtmltopdf $temp_html $pdf_path`)
    catch e
        @warn "No se pudo generar PDF. Instala wkhtmltopdf para habilitar esta función. Guardando como HTML en su lugar."
        pdf_path = output_path * ".html"
        cp(temp_html, pdf_path, force=true)
    end
    
    # Limpiar archivo temporal
    rm(temp_html, force=true)
    
    return pdf_path
end

function export_to_latex(report_data::Dict{String,Any}, template::ReportTemplate, output_path::String)
    latex_content = generate_latex_report(report_data, template)
    
    filename = output_path * ".tex"
    open(filename, "w") do io
        write(io, latex_content)
    end
    
    return filename
end

function create_default_template()
    return ReportTemplate()
end
# =============================================================================
# HTML Report Generation
# =============================================================================

function generate_html_report(data::Dict{String,Any}, template::ReportTemplate)
    # Template HTML con Mustache
    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>{{header}}</title>
        {{&css_style}}
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    </head>
    <body>
        <div class="report-container">
            <h1>{{header}}</h1>
            <p>Generated on: {{timestamp}}</p>
            
            {{#sections}}
            {{&section_content}}
            {{/sections}}
        </div>
        
        {{&javascript_code}}
    </body>
    </html>
    """
    
    # Preparar datos para el template
    template_data = Dict{String,Any}(
        "header" => template.header,
        "css_style" => template.css_style,
        "timestamp" => Dates.format(data["timestamp"], "yyyy-mm-dd HH:MM:SS"),
        "sections" => [],
        "javascript_code" => ""
    )
    
    # Generar secciones
    sections_html = []
    js_code = []
    
    for section in template.sections
        if section == :executive_summary
            push!(sections_html, generate_executive_summary_html(data["executive_summary"]))
        elseif section == :model_architecture
            push!(sections_html, generate_architecture_html(data["model_architecture"]))
        elseif section == :training_progress
            html, js = generate_training_progress_html(data["training_progress"])
            push!(sections_html, html)
            push!(js_code, js)
        elseif section == :hyperparameters
            push!(sections_html, generate_hyperparameters_html(data["config"]))
        elseif section == :performance_analysis
            push!(sections_html, generate_performance_html(data["performance_analysis"]))
        elseif section == :hardware_utilization
            push!(sections_html, generate_hardware_html(data["hardware_utilization"]))
        elseif section == :reproducibility_info
            push!(sections_html, generate_reproducibility_html(data["reproducibility"]))
        end
    end
    
    template_data["sections"] = [Dict("section_content" => html) for html in sections_html]
    template_data["javascript_code"] = "<script>\n" * join(js_code, "\n") * "\n</script>"
    
    # Renderizar template
    return Mustache.render(html_template, template_data)
end

function generate_executive_summary_html(summary::Dict{String,Any})
    html = """
    <section id="executive-summary">
        <h2>Executive Summary</h2>
        <div class="metrics-grid">
    """
    
    # Métricas principales
    metrics = [
        ("Best Training Loss", summary["best_train_loss"], "📉"),
        ("Best Validation Loss", summary["best_val_loss"], "📊"),
        ("Final Training Loss", summary["final_train_loss"], "🎯"),
        ("Final Validation Loss", summary["final_val_loss"], "🎯"),
        ("Total Training Time", "$(round(summary["total_time"]/60, digits=2)) min", "⏱️"),
        ("Avg Time per Epoch", "$(round(summary["avg_time_per_epoch"], digits=2)) sec", "⚡")
    ]
    
    for (label, value, icon) in metrics
        if isa(value, String) || (isa(value, Number) && !isnan(value))
            html *= """
            <div class="metric-box">
                <div class="metric-icon">$icon</div>
                <div class="metric-label">$label</div>
                <div class="metric-value">$(isa(value, Number) ? round(value, digits=4) : value)</div>
            </div>
            """
        end
    end
    
    html *= "</div>"
    
    # Advertencias
    if summary["potential_overfitting"]
        html *= """
        <div class="warning">
            ⚠️ Potential overfitting detected: validation loss is increasing while training loss decreases.
        </div>
        """
    end
    
    html *= "</section>"
    return html
end

function generate_architecture_html(arch_info::Dict{String,Any})
    html = """
    <section id="model-architecture">
        <h2>Model Architecture</h2>
        <div class="architecture-summary">
            <p><strong>Model Type:</strong> $(arch_info["type"])</p>
            <p><strong>Total Parameters:</strong> $(format_number(arch_info["total_parameters"]))</p>
            <p><strong>Trainable Parameters:</strong> $(format_number(arch_info["trainable_parameters"]))</p>
        </div>
    """
    
    # Tabla de capas
    if haskey(arch_info, "layers")
        html *= """
        <h3>Layer Details</h3>
        <table>
            <thead>
                <tr>
                    <th>Layer #</th>
                    <th>Type</th>
                    <th>Output Shape</th>
                    <th>Parameters</th>
                    <th>Details</th>
                </tr>
            </thead>
            <tbody>
        """
        
        for layer in arch_info["layers"]
            details = ""
            if layer["name"] == "Dense"
                details = "$(layer["input_size"]) → $(layer["output_size"])"
            elseif layer["name"] == "Activation"
                details = layer["function"]
            end
            
            html *= """
            <tr>
                <td>$(layer["index"])</td>
                <td>$(layer["name"])</td>
                <td>-</td>
                <td>$(format_number(layer["parameters"]))</td>
                <td>$details</td>
            </tr>
            """
        end
        
        html *= """
            </tbody>
        </table>
        """
    end
    
    html *= "</section>"
    return html
end

function generate_training_progress_html(progress::Dict{String,Any})
    html = """
    <section id="training-progress">
        <h2>Training Progress</h2>
        <div id="loss-plot" style="width: 100%; height: 500px;"></div>
    """
    
    # Agregar plots de métricas si existen
    if !isempty(progress["train_metrics"])
        html *= """<div id="metrics-plot" style="width: 100%; height: 500px;"></div>"""
    end
    
    html *= "</section>"
    
    # Generar JavaScript para plots interactivos
    js = generate_plotly_loss_chart(progress)
    
    return html, js
end

function generate_plotly_loss_chart(progress::Dict{String,Any})
    epochs = 1:length(progress["train_loss"])
    
    js = """
    var trace1 = {
        x: [$(join(epochs, ","))],
        y: [$(join(progress["train_loss"], ","))],
        type: 'scatter',
        name: 'Training Loss',
        line: {color: '#3498db', width: 2}
    };
    
    var data = [trace1];
    """
    
    # Agregar validation loss si existe
    if !isempty(progress["val_loss"])
        val_epochs = [i for (i, v) in enumerate(progress["val_loss"]) if !isnan(v)]
        val_values = [v for v in progress["val_loss"] if !isnan(v)]
        
        if !isempty(val_epochs)
            js *= """
            var trace2 = {
                x: [$(join(val_epochs, ","))],
                y: [$(join(val_values, ","))],
                type: 'scatter',
                name: 'Validation Loss',
                line: {color: '#e74c3c', width: 2}
            };
            data.push(trace2);
            """
        end
    end
    
    js *= """
    var layout = {
        title: 'Training and Validation Loss',
        xaxis: {title: 'Epoch'},
        yaxis: {title: 'Loss', type: 'log'},
        hovermode: 'x unified',
        showlegend: true,
        legend: {x: 0.7, y: 1}
    };
    
    Plotly.newPlot('loss-plot', data, layout);
    """
    
    # Agregar gráfico de métricas si existen
    if !isempty(progress["train_metrics"])
        js *= generate_metrics_plot_js(progress)
    end
    
    return js
end

function generate_metrics_plot_js(progress::Dict{String,Any})
    js = """
    var metricsData = [];
    """
    
    # Agregar cada métrica
    for (metric_name, values) in progress["train_metrics"]
        if !isempty(values)
            epochs = 1:length(values)
            js *= """
            metricsData.push({
                x: [$(join(epochs, ","))],
                y: [$(join(values, ","))],
                type: 'scatter',
                name: 'Train $metric_name'
            });
            """
        end
    end
    
    for (metric_name, values) in progress["val_metrics"]
        clean_indices = [i for (i, v) in enumerate(values) if !isnan(v)]
        clean_values = [v for v in values if !isnan(v)]
        
        if !isempty(clean_values)
            js *= """
            metricsData.push({
                x: [$(join(clean_indices, ","))],
                y: [$(join(clean_values, ","))],
                type: 'scatter',
                name: 'Val $metric_name'
            });
            """
        end
    end
    
    js *= """
    var metricsLayout = {
        title: 'Training Metrics',
        xaxis: {title: 'Epoch'},
        yaxis: {title: 'Value'},
        hovermode: 'x unified'
    };
    
    Plotly.newPlot('metrics-plot', metricsData, metricsLayout);
    """
    
    return js
end

function generate_hyperparameters_html(config::Dict{String,Any})
    html = """
    <section id="hyperparameters">
        <h2>Hyperparameters</h2>
        <table>
            <thead>
                <tr>
                    <th>Parameter</th>
                    <th>Value</th>
                </tr>
            </thead>
            <tbody>
    """
    
    # Ordenar parámetros para presentación consistente
    params = sort(collect(config), by=x->string(x[1]))
    
    for (key, value) in params
        # Formatear valor según tipo
        formatted_value = format_config_value(value)
        html *= """
        <tr>
            <td><code>$key</code></td>
            <td>$formatted_value</td>
        </tr>
        """
    end
    
    html *= """
            </tbody>
        </table>
    </section>
    """
    
    return html
end

function generate_performance_html(analysis::Dict{String,Any})
    html = """
    <section id="performance-analysis">
        <h2>Performance Analysis</h2>
    """
    
    if haskey(analysis, "best_epoch")
        html *= """
        <div class="success">
            🏆 Best performance achieved at epoch $(analysis["best_epoch"])
        </div>
        """
    end
    
    # Tabla detallada por época (primeras y últimas 5)
    epochs_data = analysis["epochs"]
    if length(epochs_data) > 10
        show_epochs = vcat(epochs_data[1:5], epochs_data[end-4:end])
        html *= "<p><em>Showing first 5 and last 5 epochs</em></p>"
    else
        show_epochs = epochs_data
    end
    
    html *= """
    <table>
        <thead>
            <tr>
                <th>Epoch</th>
                <th>Train Loss</th>
                <th>Val Loss</th>
    """
    
    # Agregar columnas para métricas
    if !isempty(show_epochs)
        for key in keys(show_epochs[1])
            if startswith(string(key), "train_") && key != :train_loss
                metric = replace(string(key), "train_" => "")
                html *= "<th>Train $metric</th>"
            end
        end
    end
    
    html *= """
            </tr>
        </thead>
        <tbody>
    """
    
    for epoch_data in show_epochs
        html *= "<tr>"
        html *= "<td>$(epoch_data["epoch"])</td>"
        html *= "<td>$(round(get(epoch_data, "train_loss", 0.0), digits=4))</td>"        
        val_loss = get(epoch_data, "val_loss", NaN)
        html *= "<td>$(isnan(val_loss) ? "-" : round(val_loss, digits=4))</td>"
        
        # Agregar métricas
        for key in keys(epoch_data)
            if startswith(string(key), "train_") && key != "train_loss"
                value = get(epoch_data, key, 0.0)
                html *= "<td>$(round(value, digits=4))</td>"
            end
        end
        
        html *= "</tr>"
    end
    
    html *= """
            </tbody>
        </table>
    </section>
    """
    
    return html
end

function generate_hardware_html(hardware::Dict{String,Any})
    html = """
    <section id="hardware-utilization">
        <h2>Hardware Utilization</h2>
        <div class="hardware-info">
    """
    
    # CPU Info
    html *= """
    <h3>CPU Information</h3>
    <p><strong>Threads:</strong> $(hardware["cpu_threads"])</p>
    """
    
    # GPU Info
    if hardware["gpu_available"]
        html *= """
        <h3>GPU Information</h3>
        <p><strong>Device:</strong> $(hardware["gpu_name"])</p>
        <p><strong>Total Memory:</strong> $(round(hardware["gpu_memory_total"], digits=2)) GB</p>
        <p><strong>Used Memory:</strong> $(round(hardware["gpu_memory_used"], digits=2)) GB</p>
        <p><strong>Free Memory:</strong> $(round(hardware["gpu_memory_free"], digits=2)) GB</p>
        
        <div class="metric-box">
            <div class="metric-label">GPU Memory Usage</div>
            <div class="metric-value">$(round(hardware["gpu_memory_used"] / hardware["gpu_memory_total"] * 100, digits=1))%</div>
        </div>
        """
    else
        html *= """
        <h3>GPU Information</h3>
        <p>No GPU detected - Training performed on CPU</p>
        """
    end
    
    html *= """
        </div>
    </section>
    """
    
    return html
end

function generate_reproducibility_html(repro::Dict{String,Any})
    html = """
    <section id="reproducibility">
        <h2>Reproducibility Information</h2>
        
        <h3>Environment</h3>
        <ul>
            <li><strong>Julia Version:</strong> $(repro["julia_version"])</li>
            <li><strong>Training Date:</strong> $(repro["training_date"])</li>
            <li><strong>Random Seed:</strong> $(repro["random_seed"])</li>
        </ul>
        
        <h3>Training Configuration</h3>
        <ul>
            <li><strong>Batch Size:</strong> $(repro["batch_size"])</li>
            <li><strong>Learning Rate:</strong> $(repro["learning_rate"])</li>
            <li><strong>Optimizer:</strong> $(repro["optimizer"])</li>
            <li><strong>Epochs:</strong> $(repro["epochs"])</li>
        </ul>
        
        <h3>Code to Reproduce</h3>
        <div class="code-block">
            <pre>
# Load the model and configuration
using DeepDynamics

# Set random seed
        Random.seed!($(repro["random_seed"] == "not set" ? 42 : repro["random_seed"]))

# Training configuration
config = Dict(
    "batch_size" => $(repro["batch_size"]),
    "learning_rate" => $(repro["learning_rate"]),
    "optimizer" => "$(repro["optimizer"])",
    "epochs" => $(repro["epochs"])
)

# Train model
history = fit!(model, X_train, y_train; config...)
            </pre>
        </div>
    </section>
    """
    
    return html
end

# =============================================================================
# Markdown Report Generation
# =============================================================================

function generate_markdown_report(data::Dict{String,Any}, template::ReportTemplate)
    md = """
    # $(template.header)
    
    Generated on: $(Dates.format(data["timestamp"], "yyyy-mm-dd HH:MM:SS"))
    
    """
    
    for section in template.sections
        if section == :executive_summary
            md *= generate_executive_summary_md(data["executive_summary"])
        elseif section == :model_architecture
            md *= generate_architecture_md(data["model_architecture"])
        elseif section == :training_progress
            md *= generate_training_progress_md(data["training_progress"])
        elseif section == :hyperparameters
            md *= generate_hyperparameters_md(data["config"])
        elseif section == :performance_analysis
            md *= generate_performance_md(data["performance_analysis"])
        elseif section == :hardware_utilization
            md *= generate_hardware_md(data["hardware_utilization"])
        elseif section == :reproducibility_info
            md *= generate_reproducibility_md(data["reproducibility"])
        end
        md *= "\n\n"
    end
    
    return md
end

function generate_executive_summary_md(summary::Dict{String,Any})
    md = """
    ## Executive Summary
    
    | Metric | Value |
    |--------|-------|
    | Best Training Loss | $(round(summary["best_train_loss"], digits=4)) |
    | Best Validation Loss | $(isnan(summary["best_val_loss"]) ? "N/A" : round(summary["best_val_loss"], digits=4)) |
    | Final Training Loss | $(round(summary["final_train_loss"], digits=4)) |
    | Final Validation Loss | $(isnan(summary["final_val_loss"]) ? "N/A" : round(summary["final_val_loss"], digits=4)) |
    | Total Training Time | $(round(summary["total_time"]/60, digits=2)) minutes |
    | Avg Time per Epoch | $(round(summary["avg_time_per_epoch"], digits=2)) seconds |
    """
    
    if summary["potential_overfitting"]
        md *= """
        
        > ⚠️ **Warning**: Potential overfitting detected - validation loss is increasing while training loss decreases.
        """
    end
    
    return md
end

function generate_architecture_md(arch_info::Dict{String,Any})
    md = """
    ## Model Architecture
    
    - **Model Type**: $(arch_info["type"])
    - **Total Parameters**: $(format_number(arch_info["total_parameters"]))
    - **Trainable Parameters**: $(format_number(arch_info["trainable_parameters"]))
    """
    
    if haskey(arch_info, "layers")
        md *= """
        
        ### Layer Details
        
        | Layer | Type | Parameters | Details |
        |-------|------|------------|---------|
        """
        
        for layer in arch_info["layers"]
            details = ""
            if layer["name"] == "Dense"
                details = "$(layer["input_size"]) → $(layer["output_size"])"
            elseif layer["name"] == "Activation"
                details = layer["function"]
            end
            
            md *= "| $(layer["index"]) | $(layer["name"]) | $(format_number(layer["parameters"])) | $details |\n"
        end
    end
    
    return md
end

function generate_training_progress_md(progress::Dict{String,Any})
    md = """
    ## Training Progress
    
    ### Loss Evolution
    
    Training loss decreased from $(round(progress["train_loss"][1], digits=4)) to $(round(progress["train_loss"][end], digits=4))
    """
    
    if haskey(progress, "loss_reduction")
        md *= " ($(round(progress["loss_reduction"], digits=1))% reduction)"
    end
    
    return md
end

function generate_hyperparameters_md(config::Dict{String,Any})
    md = """
    ## Hyperparameters
    
    | Parameter | Value |
    |-----------|-------|
    """
    
    for (key, value) in sort(collect(config), by=x->string(x[1]))
        md *= "| `$key` | $(format_config_value(value)) |\n"
    end
    
    return md
end

function generate_performance_md(analysis::Dict{String,Any})
    md = """
    ## Performance Analysis
    """
    
    if haskey(analysis, "best_epoch")
        md *= """
        
        Best performance achieved at epoch **$(analysis["best_epoch"])**
        """
    end
    
    return md
end

function generate_hardware_md(hardware::Dict{String,Any})
    md = """
    ## Hardware Utilization
    
    ### CPU
    - Threads: $(hardware["cpu_threads"])
    """
    
    if hardware["gpu_available"]
        md *= """
        
        ### GPU
        - Device: $(hardware["gpu_name"])
        - Total Memory: $(round(hardware["gpu_memory_total"], digits=2)) GB
        - Used Memory: $(round(hardware["gpu_memory_used"], digits=2)) GB
        - Memory Usage: $(round(hardware["gpu_memory_used"] / hardware["gpu_memory_total"] * 100, digits=1))%
        """
    else
        md *= """
        
        ### GPU
        - No GPU detected - Training performed on CPU
        """
    end
    
    return md
end

function generate_reproducibility_md(repro::Dict{String,Any})
    md = """
    ## Reproducibility Information
    
    ### Environment
    - Julia Version: $(repro["julia_version"])
    - Training Date: $(repro["training_date"])
    - Random Seed: $(repro["random_seed"])
    
    ### Training Configuration
    - Batch Size: $(repro["batch_size"])
    - Learning Rate: $(repro["learning_rate"])
    - Optimizer: $(repro["optimizer"])
    - Epochs: $(repro["epochs"])
    
    ### Code to Reproduce
    
    ```julia
    using DeepDynamics
    Random.seed!($(repro["random_seed"] == "not set" ? 42 : repro["random_seed"]))
    
    config = Dict(
        "batch_size" => $(repro["batch_size"]),
        "learning_rate" => $(repro["learning_rate"]),
        "optimizer" => "$(repro["optimizer"])",
        "epochs" => $(repro["epochs"])
    )
    
    history = fit!(model, X_train, y_train; config...)
    ```
    """
    
    return md
end

# =============================================================================
# LaTeX Report Generation
# =============================================================================

function generate_latex_report(data::Dict{String,Any}, template::ReportTemplate)
    latex = """
    \\documentclass[11pt]{article}
    \\usepackage[utf8]{inputenc}
    \\usepackage{graphicx}
    \\usepackage{booktabs}
    \\usepackage{hyperref}
    \\usepackage{amsmath}
    \\usepackage{listings}
    \\usepackage{xcolor}
    
    \\title{$(template.header)}
    \\author{DeepDynamics.jl}
    \\date{$(Dates.format(data["timestamp"], "yyyy-mm-dd"))}
    
    \\begin{document}
    \\maketitle
    
    """
    
    for section in template.sections
        if section == :executive_summary
            latex *= generate_executive_summary_latex(data["executive_summary"])
        elseif section == :model_architecture
            latex *= generate_architecture_latex(data["model_architecture"])
        # ... más secciones
        end
    end
    
    latex *= """
    \\end{document}
    """
    
    return latex
end

function generate_executive_summary_latex(summary::Dict{String,Any})
    latex = """
    \\section{Executive Summary}
    
    \\begin{table}[h]
    \\centering
    \\begin{tabular}{lr}
    \\toprule
    Metric & Value \\\\
    \\midrule
    Best Training Loss & $(round(summary["best_train_loss"], digits=4)) \\\\
    Best Validation Loss & $(isnan(summary["best_val_loss"]) ? "N/A" : round(summary["best_val_loss"], digits=4)) \\\\
    Final Training Loss & $(round(summary["final_train_loss"], digits=4)) \\\\
    Final Validation Loss & $(isnan(summary["final_val_loss"]) ? "N/A" : round(summary["final_val_loss"], digits=4)) \\\\
    Total Training Time & $(round(summary["total_time"]/60, digits=2)) min \\\\
    \\bottomrule
    \\end{tabular}
    \\end{table}
    """
    
    return latex
end

function generate_architecture_latex(arch_info::Dict{String,Any})
    latex = """
    \\section{Model Architecture}
    
    The model has $(format_number(arch_info["total_parameters"])) total parameters, 
    of which $(format_number(arch_info["trainable_parameters"])) are trainable.
    """
    
    return latex
end

# =============================================================================
# Report Callback
# =============================================================================

mutable struct ReportCallback <: AbstractCallback
    output_format::Symbol
    save_path::String
    template::ReportTemplate
    generate_intermediate::Bool
    intermediate_frequency::Int
    config::Dict{String,Any}
    history::Union{Nothing, History}  # Guardar referencia a la historia
    
    function ReportCallback(;
        output_format::Symbol = :html,
        save_path::String = "training_report",
        template::ReportTemplate = ReportTemplate(),
        generate_intermediate::Bool = false,
        intermediate_frequency::Int = 10
    )
        new(output_format, save_path, template, generate_intermediate, 
            intermediate_frequency, Dict{String,Any}(), nothing)
    end
end

function on_train_begin(cb::ReportCallback, logs::Dict)
    # Capturar configuración inicial
    cb.config["start_time"] = now()
    cb.config["optimizer"] = string(typeof(logs[:optimizer]))
    
    # Limpiar historia previa
    cb.history = nothing
    
    # Extraer hiperparámetros del optimizador si es posible
    if hasfield(typeof(logs[:optimizer]), :learning_rate)
        cb.config["learning_rate"] = logs[:optimizer].learning_rate
    end
end

function on_epoch_end(cb::ReportCallback, epoch::Int, logs::Dict)
    # Actualizar historia guardada si está disponible
    if haskey(logs, :history)
        cb.history = logs[:history]
    elseif cb.history === nothing
        # Crear historia desde logs si es posible
        cb.history = History()
        cb.history.epochs = epoch
    end
    
    # Generar reporte intermedio si está configurado
    if cb.generate_intermediate && epoch % cb.intermediate_frequency == 0
        model = logs[:model]
        
        # Usar historia guardada o crear una parcial
        partial_history = cb.history !== nothing ? cb.history : History()
        
        # Actualizar con datos actuales si están disponibles
        if haskey(logs, :loss) && (isempty(partial_history.train_loss) || partial_history.epochs < epoch)
            push!(partial_history.train_loss, Float32(logs[:loss]))
        end
        if haskey(logs, :val_loss) && (isempty(partial_history.val_loss) || partial_history.epochs < epoch)
            push!(partial_history.val_loss, Float32(logs[:val_loss]))
        end
        partial_history.epochs = epoch
        
        # Generar reporte intermedio
        intermediate_path = "$(cb.save_path)_epoch_$(epoch)"
        generate_training_report(
            model, 
            partial_history,
            cb.config;
            format=cb.output_format,
            template=cb.template,
            save_path=intermediate_path
        )
    end
end


function on_train_end(cb::ReportCallback, logs::Dict{Symbol,Any})
    # 1. Calcular tiempo total de entrenamiento (en segundos)
    cb.config["training_time"] = Dates.value(now() - cb.config["start_time"]) / 1000

    # 2. Obtener el modelo
    model = logs[:model]

    # 3. Recuperar la historia; si no existe en logs, usar la que guardó el callback
    history = get(logs, :history, cb.history)
    if history === nothing
        history = History()
        if haskey(logs, :train_losses)
            history.train_loss = Float32.(logs[:train_losses])
            history.epochs     = length(history.train_loss)
        end
        if haskey(logs, :val_losses)
            history.val_loss = Float32.(logs[:val_losses])
        end
    end

    # 4. Actualizar config final con batch_size y epochs si están presentes
    if haskey(logs, :batch_size)
        cb.config["batch_size"] = logs[:batch_size]
    end
    if haskey(logs, :epochs)
        cb.config["epochs"] = logs[:epochs]
    end

    # 5. Generar el reporte final
    generate_training_report(
        model,
        history,
        cb.config;
        format    = cb.output_format,
        template  = cb.template,
        save_path = cb.save_path
    )
end




end # module