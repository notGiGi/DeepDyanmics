module Visualizations

using Plots

export plot_training_progress, plot_metrics, plot_conv_filters

function plot_training_progress(train_losses::Vector{Float64}, val_losses::Union{Nothing, Vector{Float64}}=nothing)
    p = plot(1:length(train_losses), train_losses, label="Training Loss", xlabel="Epoch", ylabel="Loss",
             title="Training Progress", legend=:topright)
    if val_losses !== nothing
        plot!(1:length(val_losses), val_losses, label="Validation Loss")
    end
    display(p)
end

function plot_metrics(train_losses::Vector{Float64}, val_losses::Vector{Float64})
    p = plot(1:length(train_losses), train_losses, label="Training Loss", xlabel="Epoch", ylabel="Loss",
         title="Training vs Validation Loss", legend=:topright)
    plot!(1:length(val_losses), val_losses, label="Validation Loss")
    display(p)
end

function plot_conv_filters(filters::Array{Float64,4})
    kH, kW, in_channels, out_channels = size(filters)
    num_plots = out_channels
    p = plot(layout = (ceil(Int, sqrt(num_plots)), ceil(Int, sqrt(num_plots))), legend=false)
    for i in 1:out_channels
        filter_img = filters[:, :, 1, i]
        heatmap!(p[i], filter_img, title="Filter $i", colorbar=false)
    end
    display(p)
end

end  # End of module Visualizations
