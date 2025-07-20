# test_diagnostico_batchnorm.jl
using Test
using DeepDynamics
using Statistics

println("=== DIAGNÓSTICO DE BATCHNORM ===")

# Test 1: Verificar que los parámetros cambian
@testset "Diagnóstico Parámetros" begin
    # Modelo simple
    model = Sequential([
        Dense(2, 4),
        BatchNorm(4),
        Activation(relu),
        Dense(4, 2),
        Activation(softmax)
    ])
    
    # Datos simples
    X = Tensor(Float32[1.0 -1.0; 1.0 -1.0])
    y = Tensor(Float32[1.0 0.0; 0.0 1.0])
    
    opt = Adam(learning_rate=0.1f0)
    params = collect_parameters(model)
    
    # Guardar valores iniciales
    initial_values = Dict()
    for (i, p) in enumerate(params)
        initial_values[i] = copy(p.data)
    end
    
    # Verificar que BatchNorm está en params
    bn_layer = model.layers[2]
    @test bn_layer.gamma in params
    @test bn_layer.beta in params
    
    println("\nParámetros antes del entrenamiento:")
    println("Gamma: ", bn_layer.gamma.data)
    println("Beta: ", bn_layer.beta.data)
    
    # Un paso de entrenamiento
    for p in params
        zero_grad!(p)
    end
    
    # Forward
    pred = model(X)
    println("\nPredicciones: ", pred.data)
    
    # Loss
    loss = categorical_crossentropy(pred, y)
    println("Loss: ", loss.data[1])
    
    # Backward
    backward(loss, [1.0f0])
    
    # Verificar gradientes
    println("\nGradientes:")
    for (i, p) in enumerate(params)
        if p.grad !== nothing
            grad_norm = norm(p.grad.data)
            println("  Param $i grad norm: $grad_norm")
            if i <= 2  # Dense layer weights and bias
                println("    Grad sample: ", p.grad.data[1:min(3, end)]...)
            end
        else
            println("  Param $i: NO GRADIENT!")
        end
    end
    
    # Verificar específicamente BatchNorm
    println("\nBatchNorm gradientes:")
    println("  Gamma grad: ", bn_layer.gamma.grad !== nothing ? bn_layer.gamma.grad.data : "NO GRAD")
    println("  Beta grad: ", bn_layer.beta.grad !== nothing ? bn_layer.beta.grad.data : "NO GRAD")
    
    # Update
    optim_step!(opt, params)
    
    # Verificar cambios
    println("\nCambios después de update:")
    changes = false
    for (i, p) in enumerate(params)
        if !all(p.data .== initial_values[i])
            changes = true
            println("  Param $i cambió")
            if p === bn_layer.gamma
                println("    Gamma nuevo: ", p.data)
            elseif p === bn_layer.beta
                println("    Beta nuevo: ", p.data)
            end
        else
            println("  Param $i NO CAMBIÓ")
        end
    end
    
    @test changes "Ningún parámetro cambió!"
end

# Test 2: Verificar forward pass de BatchNorm
@testset "BatchNorm Forward Debug" begin
    bn = BatchNorm(2)
    
    # Input simple
    x = Tensor(Float32[1.0 2.0; 3.0 4.0]; requires_grad=true)
    
    println("\n\nBatchNorm Forward Debug:")
    println("Input: ", x.data)
    println("Input requires_grad: ", x.requires_grad)
    println("Gamma requires_grad: ", bn.gamma.requires_grad)
    println("Beta requires_grad: ", bn.beta.requires_grad)
    
    # Forward
    y = bn(x)
    
    println("\nOutput: ", y.data)
    println("Output requires_grad: ", y.requires_grad)
    println("Output has backward_fn: ", y.backward_fn !== nothing)
    
    # Test backward
    if y.backward_fn !== nothing
        # Zero grads
        zero_grad!(x)
        zero_grad!(bn.gamma)
        zero_grad!(bn.beta)
        
        # Backward con gradiente simple
        y.backward_fn(ones(Float32, size(y.data)))
        
        println("\nDespués de backward:")
        println("Input grad: ", x.grad !== nothing ? x.grad.data : "NO GRAD")
        println("Gamma grad: ", bn.gamma.grad !== nothing ? bn.gamma.grad.data : "NO GRAD")
        println("Beta grad: ", bn.beta.grad !== nothing ? bn.beta.grad.data : "NO GRAD")
        
        @test x.grad !== nothing
        @test bn.gamma.grad !== nothing
        @test bn.beta.grad !== nothing
    else
        @test false "No backward function!"
    end
end

# Test 3: Problema mínimo sin BatchNorm vs con BatchNorm
@testset "Comparación Sin/Con BatchNorm" begin
    # Datos XOR-like
    X = Tensor(Float32[
        0.0 0.0 1.0 1.0;
        0.0 1.0 0.0 1.0
    ])
    y = Tensor(Float32[
        1.0 0.0 0.0 1.0;
        0.0 1.0 1.0 0.0
    ])
    
    # Modelo sin BatchNorm
    model1 = Sequential([
        Dense(2, 8),
        Activation(relu),
        Dense(8, 2),
        Activation(softmax)
    ])
    
    # Modelo con BatchNorm
    model2 = Sequential([
        Dense(2, 8),
        BatchNorm(8),
        Activation(relu),
        Dense(8, 2),
        Activation(softmax)
    ])
    
    # Entrenar ambos
    for (name, model) in [("Sin BN", model1), ("Con BN", model2)]
        opt = Adam(learning_rate=0.1f0)
        params = collect_parameters(model)
        
        losses = Float32[]
        for epoch in 1:50
            for p in params
                zero_grad!(p)
            end
            
            pred = model(X)
            loss = categorical_crossentropy(pred, y)
            push!(losses, loss.data[1])
            
            backward(loss, [1.0f0])
            optim_step!(opt, params)
        end
        
        println("\n$name - Loss inicial: $(losses[1]), Loss final: $(losses[end])")
        println("Reducción: $((1 - losses[end]/losses[1]) * 100)%")
    end
end

println("\n=== FIN DIAGNÓSTICO ===")