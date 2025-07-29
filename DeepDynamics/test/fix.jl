# fix_all_tests.jl
# Script para arreglar TODOS los tests de una vez

# Lista de archivos que necesitan arreglos
test_files = [
    "test_integration_phase1_2.jl",
    "test_phase3_integration.jl", 
    "integration_test.jl",
    "test_phase4.jl",
    "test_phase4_full.jl",
    "test_phase5.jl",
    "test_phase6.jl"
]

# Función para arreglar un archivo
function fix_test_file(filepath)
    println("Arreglando: $filepath")
    
    # Leer el contenido
    content = read(filepath, String)
    
    # 1. Agregar import de forward si no existe
    if !occursin("import DeepDynamics: forward", content)
        # Agregar después del using DeepDynamics
        content = replace(content, 
            r"using DeepDynamics\n" => 
            "using DeepDynamics\nimport DeepDynamics: forward\n"
        )
    end
    
    # 2. Reemplazar Activation(relu) por x -> relu(x)
    content = replace(content, r"Activation\(relu\)" => "x -> relu(x)")
    content = replace(content, r"Activation\(sigmoid\)" => "x -> sigmoid(x)")
    content = replace(content, r"Activation\(tanh\)" => "x -> tanh(x)")
    
    # 3. Arreglar Dense con activation
    # Dense(in, out, activation=relu) -> Dense(in, out), x -> relu(x)
    content = replace(content, 
        r"Dense\((\d+),\s*(\d+),\s*activation\s*=\s*relu\)" => 
        s"Dense(\1, \2),\n            x -> relu(x)"
    )
    
    content = replace(content, 
        r"Dense\((\d+),\s*(\d+),\s*activation\s*=\s*sigmoid\)" => 
        s"Dense(\1, \2),\n            x -> sigmoid(x)"
    )
    
    content = replace(content, 
        r"Dense\((\d+),\s*(\d+),\s*activation\s*=\s*tanh\)" => 
        s"Dense(\1, \2),\n            x -> tanh(x)"
    )
    
    # 4. Si hay CUDA.functional(), agregar import
    if occursin("CUDA.functional()", content) && !occursin("using CUDA", content)
        content = replace(content,
            r"using DeepDynamics\n" =>
            "using DeepDynamics\nusing CUDA\n"
        )
    end
    
    # Escribir de vuelta
    write(filepath, content)
    println("✅ Arreglado: $filepath")
end

# Arreglar todos los archivos
cd("test")  # Cambiar al directorio de tests
for file in test_files
    if isfile(file)
        fix_test_file(file)
    else
        println("⚠️  No encontrado: $file")
    end
end

println("\n✨ Todos los tests han sido actualizados!")
println("\nAhora ejecuta: include(\"test/run_all_tests.jl\")")