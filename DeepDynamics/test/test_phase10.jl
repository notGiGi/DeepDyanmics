using Test
using DeepDynamics
using Random

Random.seed!(1234)

@testset "Fase 10: Pulido Final" begin
    
    @testset "1. Detección automática de GPU" begin
        # Verificar que las funciones exportadas existen
        @test isdefined(DeepDynamics, :gpu_available)
        @test isdefined(DeepDynamics, :get_device)
        @test isdefined(DeepDynamics, :set_default_device!)
        
        # Verificar que se puede consultar disponibilidad
        gpu_status = DeepDynamics.gpu_available()
        @test gpu_status isa Bool
        
        # Verificar dispositivo actual
        device = DeepDynamics.get_device()
        @test device in ["cuda", "cpu"]
        
        # Si no hay GPU, debe ser CPU
        if !gpu_status
            @test device == "cpu"
        end
        
        println("✓ GPU disponible: $gpu_status")
        println("✓ Dispositivo actual: $device")
    end
    
    @testset "2. DataLoaders con limpieza de Channels" begin
        # Crear datos de prueba
        n_samples = 100
        img_size = (3, 32, 32)
        n_classes = 10
        
        # Generar datos sintéticos
        images = [DeepDynamics.Tensor(randn(Float32, img_size...)) for _ in 1:n_samples]
        labels = [DeepDynamics.Tensor(Float32.(zeros(n_classes))) for _ in 1:n_samples]
        
        # Asignar etiquetas aleatorias
        for i in 1:n_samples
            label_idx = rand(1:n_classes)
            labels[i].data[label_idx] = 1.0f0
        end
        
        # Test DataLoader básico
        @testset "DataLoader estándar" begin
            dl = DeepDynamics.DataLoader(images, labels, 32)
            
            batch_count = 0
            for (batch_imgs, batch_labels) in dl
                batch_count += 1
                @test length(batch_imgs) <= 32
                @test length(batch_labels) == length(batch_imgs)
            end
            
            @test batch_count == Int(ceil(n_samples / 32))
            println("✓ DataLoader procesó $batch_count batches")
        end
        
        # Test OptimizedDataLoader con limpieza
        @testset "OptimizedDataLoader con limpieza" begin
            # Crear data loader optimizado
            opt_dl = DeepDynamics.optimized_data_loader(
                images, labels, 32;
                shuffle=true, 
                to_gpu=false,  # CPU para tests
                prefetch=2
            )
            
            @test opt_dl isa DeepDynamics.DataLoaders.OptimizedDataLoader
            
            # Procesar algunos batches
            batch_count = 0
            for (batch_imgs, batch_labels) in opt_dl
                batch_count += 1
                @test size(batch_imgs.data, 1) <= 32  # Batch size
                @test size(batch_imgs.data, 2) == 3    # Channels
                
                # Limitar a 3 batches para el test
                if batch_count >= 3
                    break
                end
            end
            
            # Limpiar el data loader
            DeepDynamics.DataLoaders.cleanup_data_loader!(opt_dl)
            
            # Verificar que se limpió correctamente
            @test !isopen(opt_dl.channel)
            
            println("✓ OptimizedDataLoader procesó y limpió correctamente")
        end
    end
    
    @testset "3. EmbeddingLayer con backward diferenciable" begin
        vocab_size = 1000
        embedding_dim = 128
        seq_length = 10
        
        # Crear capa de embedding
        emb_layer = DeepDynamics.Embedding(vocab_size, embedding_dim)
        
        # Verificar dimensiones
        @test size(emb_layer.weights.data) == (embedding_dim, vocab_size)
        @test emb_layer.trainable == true
        @test emb_layer.weights.requires_grad == true
        
        @testset "Forward pass" begin
            # Crear secuencia de índices
            indices = rand(1:vocab_size, seq_length)
            
            # Forward pass
            output = emb_layer(indices)
            
            @test size(output.data) == (embedding_dim, seq_length)
            @test output.requires_grad == true
            @test output.backward_fn !== nothing
            
            println("✓ Forward pass: entrada ($seq_length,) → salida $(size(output.data))")
        end
        
        @testset "Backward pass" begin
            # Reiniciar gradientes
            DeepDynamics.zero_grad!(emb_layer.weights)
            
            # Forward
            indices = [1, 5, 10, 1]  # Nota: índice 1 aparece dos veces
            output = emb_layer(indices)
            
            # Simular gradiente de pérdida
            grad_output = DeepDynamics.Tensor(ones(Float32, embedding_dim, length(indices)))
            
            # Backward
            DeepDynamics.backward(output, grad_output)
            
            # Verificar gradientes
            @test emb_layer.weights.grad !== nothing
            @test size(emb_layer.weights.grad.data) == size(emb_layer.weights.data)
            
            # El índice 1 debe tener gradiente acumulado (aparece 2 veces)
            grad_idx1 = sum(emb_layer.weights.grad.data[:, 1])
            grad_idx5 = sum(emb_layer.weights.grad.data[:, 5])
            @test abs(grad_idx1 - 2 * embedding_dim) < 1e-5  # 2 veces
            @test abs(grad_idx5 - embedding_dim) < 1e-5      # 1 vez
            
            println("✓ Backward pass con acumulación de gradientes correcta")
        end
        
        @testset "Funciones auxiliares" begin
            # Test freeze/unfreeze
            DeepDynamics.EmbeddingLayer.freeze!(emb_layer)
            @test !emb_layer.trainable
            @test !emb_layer.weights.requires_grad
            
            DeepDynamics.EmbeddingLayer.unfreeze!(emb_layer)
            @test emb_layer.trainable
            @test emb_layer.weights.requires_grad
            
            # Test get/set embedding
            test_vec = randn(Float32, embedding_dim)
            DeepDynamics.EmbeddingLayer.set_embedding!(emb_layer, 42, test_vec)
            retrieved_vec = DeepDynamics.EmbeddingLayer.get_embedding(emb_layer, 42)
            @test all(abs.(retrieved_vec .- test_vec) .< 1e-6)
            
            println("✓ Funciones auxiliares (freeze/unfreeze, get/set) funcionando")
        end
        
        @testset "Soporte para diferentes tipos de entrada" begin
            # Float64
            indices_f64 = Float64[1.0, 2.0, 3.0]
            out_f64 = emb_layer(indices_f64)
            @test size(out_f64.data) == (embedding_dim, 3)
            
            # Float32
            indices_f32 = Float32[4.0, 5.0, 6.0]
            out_f32 = emb_layer(indices_f32)
            @test size(out_f32.data) == (embedding_dim, 3)
            
            # Tensor
            indices_tensor = DeepDynamics.Tensor(Float32[7, 8, 9])
            out_tensor = emb_layer(indices_tensor)
            @test size(out_tensor.data) == (embedding_dim, 3)
            
            println("✓ Soporte para múltiples tipos de entrada")
        end
    end
    
    # Reemplazar el test de "Integración completa" (líneas 191-227) con:

    @testset "Integración completa" begin
        # Crear un modelo simple con embedding
        vocab_size = 100
        embed_dim = 16
        seq_length = 5
        hidden_dim = 32
        num_classes = 2
        
        # Entrada de prueba
        input_indices = [1, 5, 10, 20, 50]
        
        # Opción 1: Procesar paso a paso para entender dimensiones
        emb = DeepDynamics.Embedding(vocab_size, embed_dim)
        emb_output = emb(input_indices)
        println("Embedding output shape: ", size(emb_output.data))  # (16, 5)
        
        # Flatten convierte (16, 5) a (80, 1)
        flat_layer = DeepDynamics.Flatten()
        flat_output = flat_layer(emb_output)
        println("Flatten output shape: ", size(flat_output.data))  # (80, 1)
        
        # Ahora crear el modelo completo con las dimensiones correctas
        model = DeepDynamics.Sequential([
            DeepDynamics.Embedding(vocab_size, embed_dim),
            DeepDynamics.Flatten(),
            DeepDynamics.Dense(embed_dim * seq_length, hidden_dim),  # 80 -> 32
            DeepDynamics.LayerActivation(DeepDynamics.relu),
            DeepDynamics.Dense(hidden_dim, num_classes)  # 32 -> 2
        ])
        
        # Forward pass paso a paso
        emb_out = model.layers[1](input_indices)
        flat_out = model.layers[2](emb_out)
        dense1_out = model.layers[3](flat_out)
        relu_out = model.layers[4](dense1_out)
        output = model.layers[5](relu_out)
        
        @test size(output.data) == (num_classes, 1)
        
        # Verificar que se puede hacer backward
        target = DeepDynamics.Tensor(reshape(Float32[1.0, 0.0], (2, 1)))
        loss = DeepDynamics.mse_loss(output, target)
        
        # Inicializar gradientes
        params = DeepDynamics.collect_parameters(model)
        for p in params
            DeepDynamics.zero_grad!(p)
        end
        
        # Backward
        DeepDynamics.backward(loss, [1.0f0])
        
        # Verificar que los embeddings tienen gradientes
        emb_layer = model.layers[1]
        @test emb_layer.weights.grad !== nothing
        @test any(emb_layer.weights.grad.data .!= 0)
        
        println("✓ Integración completa: modelo con embeddings entrenables")
    end

    # Agregar estos tests adicionales al final de test_phase10.jl

    @testset "Validaciones críticas adicionales" begin
        
        @testset "Memory leaks en DataLoaders" begin
            # Verificar que no hay fugas de memoria con múltiples iteraciones
            if DeepDynamics.gpu_available()
                initial_mem = DeepDynamics.gpu_memory_info().used
                
                # Crear y destruir múltiples data loaders
                for i in 1:5
                    images = [DeepDynamics.Tensor(randn(Float32, 3, 32, 32)) for _ in 1:100]
                    labels = [DeepDynamics.Tensor(randn(Float32, 10)) for _ in 1:100]
                    
                    dl = DeepDynamics.optimized_data_loader(images, labels, 32; to_gpu=true)
                    
                    # Consumir algunos batches
                    count = 0
                    for batch in dl
                        count += 1
                        count >= 2 && break
                    end
                    
                    # Limpiar
                    DeepDynamics.DataLoaders.cleanup_data_loader!(dl)
                    DeepDynamics.check_and_clear_gpu_memory()
                end
                
                final_mem = DeepDynamics.gpu_memory_info().used
                mem_increase = final_mem - initial_mem
                
                # No debe haber aumento significativo de memoria
                @test mem_increase < 0.1  # GB
                println("✓ Sin fugas de memoria detectadas (aumento: $(round(mem_increase, digits=3)) GB)")
            end
        end
        
        @testset "Embedding gradientes con vocabulario grande" begin
            # Test con vocabulario realista
            vocab_size = 50000
            embed_dim = 300
            batch_size = 64
            
            emb = DeepDynamics.Embedding(vocab_size, embed_dim)
            
            # Simular batch de secuencias
            indices = rand(1:vocab_size, 20)  # secuencia de 20 tokens
            
            # Forward
            output = emb(indices)
            
            # Simular pérdida y backward
            fake_loss = sum(output.data)
            grad_output = DeepDynamics.Tensor(ones(Float32, size(output.data)...))
            
            # Medir tiempo de backward
            t_start = time()
            DeepDynamics.backward(output, grad_output)
            t_backward = time() - t_start
            
            @test t_backward < 0.1  # Debe ser rápido
            @test emb.weights.grad !== nothing
            
            # Verificar sparsity del gradiente
            non_zero_grads = count(!=(0), emb.weights.grad.data)
            expected_non_zero = embed_dim * length(unique(indices))
            
            @test non_zero_grads == expected_non_zero
            println("✓ Gradientes sparse correctos: $non_zero_grads/$expected_non_zero elementos no-zero")
        end
        
        @testset "GPU auto-detection bajo diferentes condiciones" begin
            # Guardar estado actual
            original_device = DeepDynamics.get_device()
            
            # Test cambio de dispositivo
            if DeepDynamics.gpu_available()
                DeepDynamics.set_default_device!("cpu")
                @test DeepDynamics.get_device() == "cpu"
                
                DeepDynamics.set_default_device!("cuda")
                @test DeepDynamics.get_device() == "cuda"
            else
                # Si no hay GPU, intentar setear cuda debe mantener cpu
                DeepDynamics.set_default_device!("cuda")
                @test DeepDynamics.get_device() == "cpu"
            end
            
            # Restaurar
            DeepDynamics.set_default_device!(original_device)
        end
        
        @testset "Robustez ante errores en DataLoader" begin
            # Test con datos corruptos
            images = [DeepDynamics.Tensor(randn(Float32, 3, 32, 32)) for _ in 1:10]
            labels = [DeepDynamics.Tensor(randn(Float32, 10)) for _ in 1:5]  # Menos labels que images
            
            @test_throws AssertionError DeepDynamics.DataLoader(images, labels, 4)
            
            # Test con batch_size > datos
            small_images = [DeepDynamics.Tensor(randn(Float32, 3, 32, 32)) for _ in 1:3]
            small_labels = [DeepDynamics.Tensor(randn(Float32, 10)) for _ in 1:3]
            
            dl = DeepDynamics.DataLoader(small_images, small_labels, 10)
            batch_count = 0
            for batch in dl
                batch_count += 1
            end
            @test batch_count == 1  # Solo debe haber 1 batch
        end
        
        @testset "EmbeddingLayer edge cases" begin
            emb = DeepDynamics.Embedding(100, 16)
            
            # Test con índice 0 (padding)
            output_pad = emb([0, 1, 2])
            @test all(output_pad.data[:, 1] .== 0.0f0)  # Padding debe ser zeros
            
            # Test índice fuera de rango
            @test_throws ErrorException emb([101])  # vocab_size = 100
            @test_throws ErrorException emb([-1])
            
            # Test secuencia vacía
            @test_throws BoundsError emb(Int[])
        end
        
        # En test_phase10.jl, reemplazar todo el test "Modelo completo entrenamiento real" con esta versión corregida:

        @testset "Modelo completo entrenamiento real" begin
            # Mini test de entrenamiento real
            vocab_size = 100
            embed_dim = 8
            hidden_dim = 16
            num_classes = 2
            
            # Crear modelo
            model = DeepDynamics.Sequential([
                DeepDynamics.Embedding(vocab_size, embed_dim),
                DeepDynamics.Flatten(),
                DeepDynamics.Dense(embed_dim * 10, hidden_dim),
                DeepDynamics.LayerActivation(DeepDynamics.relu),
                DeepDynamics.Dense(hidden_dim, num_classes),
                DeepDynamics.LayerActivation(DeepDynamics.softmax)
            ])
            
            # Datos sintéticos
            n_samples = 50
            seq_length = 10
            
            # Generar secuencias aleatorias
            inputs = [rand(1:vocab_size, seq_length) for _ in 1:n_samples]
            
            # Labels binarios - IMPORTANTE: shape debe ser (2, 1) no (2,)
            targets = []
            for i in 1:n_samples
                label_vec = zeros(Float32, num_classes)
                label_vec[i % 2 + 1] = 1.0f0  # One-hot encoding
                # Reshape para que sea (2, 1) en lugar de (2,)
                push!(targets, DeepDynamics.Tensor(reshape(label_vec, (num_classes, 1))))
            end
            
            # Optimizer
            opt = DeepDynamics.Adam(learning_rate=0.01)
            params = DeepDynamics.collect_parameters(model)
            
            # Train por 5 epochs
            initial_loss = 0.0
            final_loss = 0.0
            
            for epoch in 1:5
                epoch_loss = 0.0
                
                for i in 1:n_samples
                    # Zero grad
                    for p in params
                        DeepDynamics.zero_grad!(p)
                    end
                    
                    # Forward
                    emb_out = model.layers[1](inputs[i])
                    flat_out = model.layers[2](emb_out)
                    h1 = model.layers[3](flat_out)
                    a1 = model.layers[4](h1)
                    h2 = model.layers[5](a1)
                    output = model.layers[6](h2)
                    
                    # Verificar shapes
                    @assert size(output.data) == (num_classes, 1) "Output shape incorrecto: $(size(output.data))"
                    @assert size(targets[i].data) == (num_classes, 1) "Target shape incorrecto: $(size(targets[i].data))"
                    
                    # Loss
                    loss = DeepDynamics.categorical_crossentropy(output, targets[i])
                    epoch_loss += loss.data[1]
                    
                    # Backward
                    DeepDynamics.backward(loss, DeepDynamics.Tensor([1.0f0]))
                    
                    # Update
                    DeepDynamics.optim_step!(opt, params)
                end
                
                epoch_loss /= n_samples
                if epoch == 1
                    initial_loss = epoch_loss
                elseif epoch == 5
                    final_loss = epoch_loss
                end
            end
            
            # La pérdida debe disminuir
            @test final_loss < initial_loss
            println("✓ Entrenamiento real: loss inicial=$(round(initial_loss, digits=4)) → final=$(round(final_loss, digits=4))")
        end
    end

    println("\n🔬 Tests exhaustivos completados")
    println("📊 Cobertura extendida:")
    println("- Memory leaks")
    println("- Gradientes con vocabularios grandes") 
    println("- Robustez ante errores")
    println("- Edge cases")
    println("- Entrenamiento end-to-end")
end

println("\n🎉 Fase 10 completada exitosamente!")
println("✅ Detección automática de GPU implementada")
println("✅ DataLoaders con limpieza de Channels")
println("✅ EmbeddingLayer con backward diferenciable")
println("\n📊 Resumen de mejoras:")
println("- API más robusta con detección automática de dispositivos")
println("- Gestión de memoria mejorada en DataLoaders")
println("- Embeddings completamente diferenciables para NLP")