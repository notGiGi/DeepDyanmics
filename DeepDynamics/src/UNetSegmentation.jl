module UNetSegmentation

using DeepDynamics.TensorEngine
using DeepDynamics.ConvolutionalLayers
using DeepDynamics.AbstractLayer
using DeepDynamics.Layers
using LinearAlgebra

export UNet, forward

"""
    UNet(in_channels::Int, out_channels::Int)

Creates a simple U-Net architecture for image segmentation.
- `in_channels`: number of input channels (e.g., 3 for RGB images)
- `out_channels`: number of output channels (e.g., 1 for binary segmentation)
"""
mutable struct UNet <: AbstractLayer.Layer
    conv1::ConvolutionalLayers.Conv2D
    pool1::ConvolutionalLayers.MaxPooling
    conv2::ConvolutionalLayers.Conv2D
    pool2::ConvolutionalLayers.MaxPooling
    conv3::ConvolutionalLayers.Conv2D
    up1::Function      # Upsampling function
    conv4::ConvolutionalLayers.Conv2D
    up2::Function      # Upsampling function
    conv5::ConvolutionalLayers.Conv2D
    final_conv::ConvolutionalLayers.Conv2D
end

function UNet(in_channels::Int, out_channels::Int)
    conv1 = ConvolutionalLayers.Conv2D(in_channels, 64, (3,3); stride=1, padding=1)
    pool1 = ConvolutionalLayers.MaxPooling((2,2))
    conv2 = ConvolutionalLayers.Conv2D(64, 128, (3,3); stride=1, padding=1)
    pool2 = ConvolutionalLayers.MaxPooling((2,2))
    conv3 = ConvolutionalLayers.Conv2D(128, 256, (3,3); stride=1, padding=1)
    # Upsampling se implementa repitiendo la imagen a lo largo de las dimensiones espaciales
    up1 = x -> TensorEngine.Tensor(repeat(x.data, inner=(1,2,2)))
    conv4 = ConvolutionalLayers.Conv2D(256, 128, (3,3); stride=1, padding=1)
    up2 = x -> TensorEngine.Tensor(repeat(x.data, inner=(1,2,2)))
    conv5 = ConvolutionalLayers.Conv2D(128, 64, (3,3); stride=1, padding=1)
    final_conv = ConvolutionalLayers.Conv2D(64, out_channels, (1,1); stride=1, padding=0)
    return UNet(conv1, pool1, conv2, pool2, conv3, up1, conv4, up2, conv5, final_conv)
end

function forward(model::UNet, input::TensorEngine.Tensor)
    # Encoder
    x1 = model.conv1(input)           # (H, W, 64)
    p1 = model.pool1(x1)              # (H/2, W/2, 64)
    x2 = model.conv2(p1)              # (H/2, W/2, 128)
    p2 = model.pool2(x2)              # (H/4, W/4, 128)
    x3 = model.conv3(p2)              # (H/4, W/4, 256)
    # Decoder
    u1 = model.up1(x3)                # Upsample to approx (H/2, W/2, 256)
    c4 = model.conv4(u1)              # (H/2, W/2, 128)
    u2 = model.up2(c4)                # Upsample to approx (H, W, 128)
    c5 = model.conv5(u2)              # (H, W, 64)
    out = model.final_conv(c5)        # (H, W, out_channels)
    return out
end

# Hacer que el modelo UNet sea llamable con ()
function (model::UNet)(input::TensorEngine.Tensor)
    return forward(model, input)
end

end  # module UNetSegmentation
