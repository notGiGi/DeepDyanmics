module ImageProcessing

using Images
using FileIO
using Glob
using DeepDynamics.TensorEngine: Tensor
using Random
using ImageTransformations
using CUDA: CuArray, CUDA
#using Base.Threads  # Añadir esta línea al inicio del módulo
#using Base: @kwdef

export load_image, load_images_from_folder, prepare_batch, augment_image

# ==================================================================
# Configuración Interna
# ==================================================================

const DEFAULT_TARGET_SIZE = (64, 64)
const DEFAULT_DTYPE = Float32
const DEFAULT_BATCH_SIZE = 32

# Memoria pinned para transferencias rápidas CPU → GPU
@kwdef struct PinnedArray{T,N}
    data::Array{T,N}
end

# ==================================================================
# Funciones Públicas (API del Usuario)
# ==================================================================

"""
    load_image(path::String; target_size=(64,64))

Carga una imagen desde `path`, la redimensiona al tamaño objetivo y retorna un Tensor
con datos normalizados en el rango [0,1]. El tensor tendrá forma (altura, ancho, canales).
"""
function load_image(path::String; target_size::Tuple{Int, Int}=(64, 64))
    # Cargar imagen y redimensionar
    img = Images.load(path)
    img_resized = Images.imresize(img, target_size)
    
    # Convertir a Float32 y normalizar (si es necesario)
    img_float = Float32.(Images.channelview(img_resized))  # HWC → CHW
    
    # Crear Tensor con el constructor correcto
    Tensor(img_float)  # Usa el constructor que convierte a Float32
end

"""
    load_images_from_folder(folder::String; target_size=(64,64))

Carga todas las imágenes del folder (y subfolders) con extensiones jpg, png o jpeg.
Retorna un vector de Tensors (cada uno representando una imagen) y un vector de sus rutas.
"""
function load_images_from_folder(folder::String; target_size=DEFAULT_TARGET_SIZE)
    files = Glob.glob("**/*.{jpg,png,jpeg}", folder)
    images = [load_image(file; target_size=target_size) for file in files]
    return images, files
end

"""
    prepare_batch(images::Vector{Tensor}, labels::Vector, batch_size=32)

Prepara batches para entrenamiento dados un vector de imágenes (Tensor) y sus etiquetas correspondientes.
Cada batch es una tupla (batch_inputs, batch_labels), donde:
 - batch_inputs es un CuArray en GPU.
 - batch_labels es un CuArray en GPU.
"""
function prepare_batch(images::Vector{Tensor}, labels::Vector, batch_size=DEFAULT_BATCH_SIZE)
    # Cargar y preprocesar en paralelo
    gpu_images, gpu_labels = _load_and_preprocess(images, labels)
    
    # Dividir en batches
    batches = []
    for i in 1:batch_size:length(images)
        batch_range = i:min(i+batch_size-1, length(images))
        push!(batches, (
            gpu_images[:, :, :, batch_range],
            gpu_labels[:, batch_range]
        ))
    end
    return batches
end

# ==================================================================
# Funciones Internas (Optimizadas)
# ==================================================================

function _load_and_preprocess(images::Vector{Tensor}, labels::Vector)
    # Preprocesar en paralelo
    pinned_images = Vector{PinnedArray{DEFAULT_DTYPE, 4}}(undef, length(images))
    pinned_labels = Vector{PinnedArray{DEFAULT_DTYPE, 2}}(undef, length(images))
    
    Threads.@threads for i in eachindex(images)
        img = images[i].data
        lbl = labels[i].data
        
        # Aumentación en CPU
        img = _augment_image(img)
        
        # Usar memoria pinned
        pinned_images[i] = PinnedArray(reshape(img, size(img)..., 1))
        pinned_labels[i] = PinnedArray(reshape(lbl, size(lbl)..., 1))
    end
    
    # Mover todo a GPU de una vez
    gpu_images = CuArray(cat([x.data for x in pinned_images]..., dims=4))
    gpu_labels = CuArray(cat([x.data for x in pinned_labels]..., dims=2))
    
    return gpu_images, gpu_labels
end

function augment_image(img::Array{DEFAULT_DTYPE, 3})
    # Volteo horizontal aleatorio
    if rand() < 0.5
        img = img[:, end:-1:1, :]
    end
    
    # Rotación aleatoria
    angle = rand() * 30 - 15  # Entre -15 y 15 grados
    img = imrotate(img, deg2rad(angle))
    
    # Ajuste de brillo
    img .*= rand(DEFAULT_DTYPE) * 0.4 + 0.8  # Factor entre 0.8 y 1.2
    return img
end

end  # module ImageProcessing