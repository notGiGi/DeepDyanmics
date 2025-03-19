module DataAugmentation

using Images
using Random

export apply_augmentation, augment_batch

"""
Aplica augmentación de datos a una imagen
"""
function apply_augmentation(image; 
                           flip_horizontal=true,
                           rotation_degrees=15,
                           brightness_adjust=0.2,
                           jitter=0.1)
    # Obtener dimensiones originales
    original_size = size(image)
    
    # Convertir a array para manipulación
    img_array = Float32.(channelview(image))
    
    # Flip horizontal con 50% de probabilidad
    if flip_horizontal && rand() > 0.5
        img_array = reverse(img_array, dims=3)  # Invertir dimensión de ancho
    end
    
    # Rotación aleatoria
    if rotation_degrees > 0 && rand() > 0.5
        angle = (rand() * 2 - 1) * rotation_degrees * (π/180)  # Convertir a radianes
        img_array = imrotate(img_array, angle)
        
        # Recortar al tamaño original
        img_array = center_crop(img_array, original_size[2:3])
    end
    
    # Ajuste de brillo
    if brightness_adjust > 0 && rand() > 0.5
        factor = 1.0 + (rand() * 2 - 1) * brightness_adjust
        img_array = clamp.(img_array .* factor, 0, 1)
    end
    
    # Jitter de color (canales RGB)
    if jitter > 0 && rand() > 0.5
        for ch in 1:min(3, size(img_array, 1))
            factor = 1.0 + (rand() * 2 - 1) * jitter
            img_array[ch, :, :] = clamp.(img_array[ch, :, :] .* factor, 0, 1)
        end
    end
    
    # Convertir de vuelta a imagen
    return colorview(RGB, img_array)
end

"""
Función auxiliar para recortar al centro
"""
function center_crop(img, target_size)
    h, w = size(img)[2:3]
    th, tw = target_size
    
    i = round(Int, (h - th) / 2) + 1
    j = round(Int, (w - tw) / 2) + 1
    
    return img[:, i:i+th-1, j:j+tw-1]
end

"""
Aplica augmentación a un batch completo
"""
function augment_batch(images; kwargs...)
    return [apply_augmentation(img; kwargs...) for img in images]
end

end # module