module Reports

# El módulo Reports original mantiene su funcionalidad básica
# pero ahora incluye el nuevo ReportGenerator como submódulo
using ..TensorEngine
using ..NeuralNetwork
using ..Training
using ..Callbacks
using ..Visualizations
using Plots
using Dates

using Plots
using Dates

# Exportar funciones del módulo original
export generate_report

# Mantener la función original para compatibilidad
function generate_report(train_losses::Vector{Float64}, val_losses::Vector{Float64}; 
                        output_format=:html, filename="report.html")
    p = plot(1:length(train_losses), train_losses, label="Training Loss", 
             xlabel="Epoch", ylabel="Loss",
             title="Training Progress", legend=:topright)
    plot!(1:length(val_losses), val_losses, label="Validation Loss")
    temp_plot_file = "temp_plot.png"
    savefig(p, temp_plot_file)
    
    report_md = """
    # DeepDynamics Training Report

    **Date:** $(Dates.format(Dates.now(), "yyyy-mm-dd HH:MM:SS"))

    ## Training Summary
    - Total epochs: $(length(train_losses))
    - Final Training Loss: $(train_losses[end])
    - Final Validation Loss: $(val_losses[end])

    ## Loss Curve
    ![]($temp_plot_file)
    """
    
    open(filename, "w") do io
        write(io, report_md)
    end
    
    println("Report generated and saved to ", filename)
end

end  # End of module Reports