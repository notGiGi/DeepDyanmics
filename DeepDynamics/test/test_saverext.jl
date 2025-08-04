# test/test_model_saver_extended.jl
using Test
using DeepDynamics
using CUDA

@testset "ModelSaver Tests - Compatibilidad y Extensiones" begin
    # Modelo de prueba
    model = Sequential([
        Dense(10, 20),
        BatchNorm(20),
        Activation(relu),
        DropoutLayer(0.5),
        Dense(20, 10),
        Activation(softmax)
    ])
    
    x = Tensor(randn(Float32, 10, 5))
    
    # ========================================
    # TESTS DE COMPATIBILIDAD (API Original)
    # ========================================
    @testset "API Original - save_model(filepath, model)" begin
        filepath = "test_original_api.jld"
        
        # Usar la API original (filepath primero)
        save_model(filepath, model; metadata=Dict("api" => "original"))
        
        # Cargar con API original
        loaded = load_model(filepath)
        
        set_training_mode!(model, false)
        set_training_mode!(loaded, false)
        
        @test isapprox(model(x).data, loaded(x).data, rtol=1e-5)
        
        rm(filepath)
        println("✅ API original funciona correctamente")
    end
    
    # ========================================
    # TESTS DE LA NUEVA API
    # ========================================
    @testset "Nueva API - save_model(model, filepath)" begin
        # Test 1: Formato JLD2 con compresión
        @testset "JLD2 Format" begin
            filepath = "test_new_api.jld2"
            
            # Usar la nueva API (model primero)
            bundle = save_model(model, filepath;  # <- Nota: model primero
                              format=:jld2,
                              compression=true,
                              metadata=Dict("api" => "nueva"))
            
            @test isfile(filepath)
            @test bundle isa ModelBundle
            @test !isempty(bundle.checksum)
            
            # Cargar con nueva API
            loaded = load_model(filepath; validate_checksum=true)
            
            set_training_mode!(model, false)
            set_training_mode!(loaded, false)
            
            @test isapprox(model(x).data, loaded(x).data, rtol=1e-5)
            
            rm(filepath)
        end
        
        # Test 2: Fallback al formato original
        @testset "Fallback to Original Format" begin
            filepath = "test_fallback.jld"
            
            # Nueva API pero con formato original
            save_model(model, filepath; 
                      format=:serialization,
                      compression=false)
            
            # Debe poder cargarse con ambas APIs
            loaded1 = load_model(filepath)  # API original
            loaded2 = load_model(filepath; device="auto")  # API nueva
            
            set_training_mode!(model, false)
            set_training_mode!(loaded1, false)
            set_training_mode!(loaded2, false)
            
            @test isapprox(model(x).data, loaded1(x).data, rtol=1e-5)
            @test isapprox(model(x).data, loaded2(x).data, rtol=1e-5)
            
            rm(filepath)
        end
        
        # Test 3: Diferentes tipos de compresión
        @testset "Compression Types" begin
            # Para modelos muy pequeños, la compresión puede aumentar el tamaño
            # debido a headers y metadata. Esto es normal y esperado.
            
            # Crear un modelo más grande para que la compresión sea efectiva
            large_model = Sequential([
                Dense(100, 200),
                Dense(200, 200),
                Dense(200, 100)
            ])
            
            # Crear datos de prueba para el modelo grande
            x_large = Tensor(randn(Float32, 100, 5))
            
            file_none = "test_no_comp_large.jld2"
            file_zstd = "test_zstd_large.jld2"
            file_lz4 = "test_lz4_large.jld2"
            
            # Guardar sin compresión
            save_model(large_model, file_none; compression=false)
            size_none = filesize(file_none)
            
            # Guardar con Zstd
            save_model(large_model, file_zstd; compression=:zstd)
            size_zstd = filesize(file_zstd)
            
            # Guardar con LZ4
            save_model(large_model, file_lz4; compression=:lz4, optimize_for=:speed)
            size_lz4 = filesize(file_lz4)
            
            # Ahora sí debería comprimir
            @test size_zstd < size_none
            @test size_lz4 < size_none
            
            # Verificar que todos cargan correctamente
            set_training_mode!(large_model, false)
            y_original = large_model(x_large)
            
            for file in [file_none, file_zstd, file_lz4]
                loaded = load_model(file)
                set_training_mode!(loaded, false)
                y_loaded = loaded(x_large)
                @test isapprox(y_original.data, y_loaded.data, rtol=1e-5)
                rm(file)
            end
        end
    end
    
    # ========================================
    # TESTS DEL MODEL REGISTRY
    # ========================================
    @testset "Model Registry" begin
        registry_dir = "test_registry"
        registry = ModelRegistry(registry_dir)
        
        # Registrar modelo v1.0
        info1 = register_model(registry, model, "test_model", ["test", "v1"];
                              version="1.0.0",
                              metadata=Dict("accuracy" => 0.95))
        
        @test info1.name == "test_model"
        @test "test" in info1.tags
        @test info1.version == "1.0.0"
        
        # Registrar modelo v1.1
        info2 = register_model(registry, model, "test_model", ["test", "v1.1"];
                              version="1.1.0",
                              metadata=Dict("accuracy" => 0.97))
        
        # Listar todos los modelos
        all_models = list_models(registry)
        @test length(all_models) == 2
        
        # Filtrar por tags
        v1_models = list_models(registry; filter_tags=["v1"])
        @test length(v1_models) == 1
        
        # Obtener versión específica
        loaded_v1, info_v1 = get_model(registry, "test_model"; version="1.0.0")
        @test info_v1.version == "1.0.0"
        
        # Obtener versión latest
        loaded_latest, info_latest = get_model(registry, "test_model")
        @test info_latest.version == "1.1.0"
        
        # Verificar que ambas funcionan
        set_training_mode!(model, false)
        set_training_mode!(loaded_v1, false)
        set_training_mode!(loaded_latest, false)
        
        @test isapprox(model(x).data, loaded_v1(x).data, rtol=1e-5)
        @test isapprox(model(x).data, loaded_latest(x).data, rtol=1e-5)
        
        rm(registry_dir, recursive=true)
    end
    
    # ========================================
    # TESTS DE CHECKPOINTS (sin cambios)
    # ========================================
    @testset "Checkpoints - Compatibilidad Total" begin
        opt = Adam(learning_rate=0.001)
        params = collect_parameters(model)
        
        # Simular entrenamiento
        for p in params
            p.grad = Tensor(randn(Float32, size(p.data)...))
        end
        optim_step!(opt, params)
        
        epoch = 10
        metrics = Dict("loss" => 0.25f0)
        
        filepath = "test_checkpoint.jld"
        
        # Las funciones de checkpoint no cambiaron
        save_checkpoint(filepath, model, opt, epoch, metrics)
        loaded_model, loaded_opt, loaded_epoch, loaded_metrics = load_checkpoint(filepath)
        
        @test loaded_epoch == epoch
        @test loaded_metrics == metrics
        @test loaded_opt.learning_rate == opt.learning_rate
        
        rm(filepath)
    end
    
    # ========================================
    # TEST DE VALIDACIÓN
    # ========================================
    # En test_saverext.jl, mejorar el test de corrupción:
    @testset "Checksum Validation" begin
        filepath = "test_checksum.jld2"
        
        # Guardar con nueva API
        bundle = save_model(model, filepath; format=:jld2)
        original_checksum = bundle.checksum
        
        # Cargar con validación (debe pasar)
        @test_nowarn load_model(filepath; validate_checksum=true)
        
        # Corromper el archivo de manera más efectiva
        # Leer el archivo y buscar donde están los weights
        bytes = read(filepath)
        
        # JLD2 guarda los datos después de los headers
        # Corromper múltiples bytes en la sección de datos (después del byte 1000)
        if length(bytes) > 2000
            for i in 1500:10:1600  # Corromper múltiples bytes
                bytes[i] = ~bytes[i]
            end
        else
            # Si el archivo es pequeño, corromper el 10%
            for i in 100:10:min(length(bytes)-10, 200)
                bytes[i] = ~bytes[i]
            end
        end
        
        write(filepath, bytes)
        
        # Debe fallar la validación
        @test_throws ErrorException load_model(filepath; validate_checksum=true)
        
        # Sin validación debe cargar pero advertir
        @test_nowarn load_model(filepath; validate_checksum=false)
        
        rm(filepath)
    end
end

# ========================================
# EJECUTAR TODOS LOS TESTS
# ========================================
println("\n🧪 Ejecutando tests de ModelSaver...")
println("=" ^ 50)

# Primero ejecutar los tests originales
println("\n📋 Tests originales del ModelSaver:")
include("test_saver.jl")  # Los tests originales

println("\n📋 Tests extendidos del ModelSaver:")
# Luego los nuevos tests

println("\n✅ ¡Todos los tests pasaron!")