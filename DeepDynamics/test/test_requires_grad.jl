using DeepDynamics

println("=== Test de requires_grad ===")

a = Tensor([1.0f0, 2.0f0]; requires_grad=true)
b = Tensor([3.0f0, 4.0f0]; requires_grad=false)
c = Tensor([5.0f0, 6.0f0]; requires_grad=true)

println("\nConfiguracion:")
println("  a.requires_grad = ", a.requires_grad)
println("  b.requires_grad = ", b.requires_grad)
println("  c.requires_grad = ", c.requires_grad)

result = add(add(a, b), c)
backward(result, Tensor([1.0f0, 1.0f0]))

println("\nResultados:")
println("  a.grad: ", a.grad !== nothing ? "✓ Existe" : "✗ No existe")
println("  b.grad: ", b.grad !== nothing ? "✓ Existe" : "✗ No existe")
println("  c.grad: ", c.grad !== nothing ? "✓ Existe" : "✗ No existe")

@assert a.grad !== nothing "a debería tener gradiente"
@assert b.grad === nothing "b NO debería tener gradiente"
@assert c.grad !== nothing "c debería tener gradiente"

println("\n✓ requires_grad funciona correctamente")
