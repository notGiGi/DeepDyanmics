# test_phase3_fixed.jl
using DeepDynamics
using Test
using CUDA

println("=== TEST FASE 3: Capas Fundamentales (CORREGIDO) ===")

@testset "FASE 3 - Capas Fundamentales" begin
    # Test 3.1: Dense verifica dimensiones
    @test_throws AssertionError begin
        layer = Dense(10, 5)
        wrong_input = Tensor(randn(Float32, 8, 2))  # 8 != 10
        layer(wrong_input)
    end
    println("✓ Dense rechaza dimensiones incorrectas")
    
    # Test 3.2: Flatten detecta formatos
    # NCHW
    x_nchw = Tensor(randn(Float32, 32, 3, 64, 64))  # batch=32
    flat = Flatten()
    y = flat(x_nchw)
    @test size(y.data) == (3*64*64, 32)  # (features, batch)
    println("✓ Flatten procesa formato NCHW")
    
    # WHCN
    x_whcn = Tensor(randn(Float32, 224, 224, 3, 8))  # batch=8
    y2 = flat(x_whcn)
    @test size(y2.data) == (224*224*3, 8)
    println("✓ Flatten procesa formato WHCN")
    
    # Test 3.3: stack_batch preserva dispositivo
    # CPU
    t1_cpu = Tensor(randn(Float32, 3, 32, 32))
    t2_cpu = Tensor(randn(Float32, 3, 32, 32))
    batch_cpu = stack_batch([t1_cpu, t2_cpu])
    @test !(batch_cpu.data isa CUDA.CuArray)
    @test size(batch_cpu.data) == (2, 3, 32, 32)
    println("✓ stack_batch funciona en CPU")
    
    if CUDA.functional()
        # GPU - todos en el mismo dispositivo
        t1_gpu = to_gpu(Tensor(randn(Float32, 3, 32, 32)))
        t2_gpu = to_gpu(Tensor(randn(Float32, 3, 32, 32)))
        batch_gpu = stack_batch([t1_gpu, t2_gpu])
        @test batch_gpu.data isa CUDA.CuArray
        println("✓ stack_batch funciona en GPU")
        
        # Mezcla - debe convertir al dispositivo del primero
        t3_cpu = Tensor(randn(Float32, 3, 32, 32))
        batch_mixed = stack_batch([t1_gpu, t3_cpu])  # Primero es GPU
        @test batch_mixed.data isa CUDA.CuArray
        println("✓ stack_batch convierte tensores mixtos")
    end
end

println("✓ TEST FASE 3 COMPLETADO")
