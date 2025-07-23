# test_phase5.jl
using Test
using DeepDynamics
using CUDA

println("=== TESTS FASE 5: Formato Consistente + Conv2D Fix ===")

@testset "Fase 5: Formato NCHW" begin
    
    # Test 5.1: Conv2D formato de entrada y bias
    @testset "Conv2D formato correcto" begin
        conv = Conv2D(3, 16, (3,3), padding=(1,1))
        
        # Verificar formato de bias
        @test size(conv.bias.data) == (1, 16, 1, 1)
        
        # Input NCHW
        x = Tensor(randn(Float32, 4, 3, 32, 32))  # (N,C,H,W)
        
        # Forward debe funcionar
        y = conv(x)
        @test size(y.data) == (4, 16, 32, 32)  # Mismo H,W por padding
        
        # Test error con formato incorrecto
        x_bad = Tensor(randn(Float32, 32, 32, 3))  # 3D
        @test_throws ErrorException conv(x_bad)
    end
    
    # Test 5.2: Fallback a NNlib cuando cuDNN falla
    @testset "Conv2D fallback" begin
        if CUDA.functional()
            # Crear Conv2D que podría fallar en cuDNN
            conv = Conv2D(3, 64, (7,7), stride=(2,2), padding=(3,3))
            x = Tensor(randn(Float32, 1, 3, 224, 224))
            
            # Debe funcionar (con cuDNN o fallback)
            y = conv(x)
            @test size(y.data) == (1, 64, 112, 112)
        else
            @test true  # Skip en CPU
        end
    end
    
    # Test 5.3: MaxPooling respeta padding
    @testset "MaxPooling padding" begin
        # Con padding
        pool1 = MaxPooling((3,3), stride=(1,1), padding=(1,1))
        x = Tensor(randn(Float32, 2, 8, 10, 10))
        y1 = pool1(x)
        @test size(y1.data) == (2, 8, 10, 10)  # Mismo tamaño por padding
        
        # Sin padding
        pool2 = MaxPooling((3,3), stride=(1,1), padding=(0,0))
        y2 = pool2(x)
        @test size(y2.data) == (2, 8, 8, 8)  # Reducido
    end
    
    # Test 5.4: ConvKernelLayer preserva NCHW
    @testset "ConvKernelLayer formato" begin
        conv = ConvKernelLayer(3, 32, (3,3), stride=(1,1), padding=(1,1))
        x = Tensor(randn(Float32, 4, 3, 28, 28))
        
        y = conv(x)
        @test size(y.data) == (4, 32, 28, 28)  # NCHW preservado
    end
    
    # Test 5.5: Integración CNN completa
    @testset "CNN integración" begin
        model = Sequential([
            Conv2D(3, 32, (3,3), padding=(1,1)),
            BatchNorm(32),
            Activation(relu),
            MaxPooling((2,2)),
            ConvKernelLayer(32, 64, (3,3), padding=(1,1)),
            BatchNorm(64),
            Activation(relu),
            MaxPooling((2,2)),
            Flatten(),
            Dense(64*8*8, 10),
            Activation(softmax)
        ])
        
        # Forward con batch
        X = Tensor(randn(Float32, 4, 3, 32, 32))
        y = model(X)
        
        @test size(y.data) == (10, 4)  # (clases, batch)
        
        # Verificar que no hay NaN
        @test !any(isnan.(y.data))
    end
end

println("\n=== TODOS LOS TESTS DE FASE 5 PASARON ===")