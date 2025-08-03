# test_report_generator.jl

using Test
using DeepDynamics
using DeepDynamics.ReportGenerator
using DeepDynamics.Training: History
using DeepDynamics.NeuralNetwork
using DeepDynamics.NeuralNetwork: Activation  # Agregar import de Activation
using DeepDynamics.Callbacks
using CUDA
using Dates
using Random

println("="^60)
println("Test de ReportGenerator.jl")
println("="^60)

# Función auxiliar para crear datos sintéticos
function create_test_data(n_samples=100, n_features=10, n_classes=3)
    X = [Tensor(randn(Float64, n_features)) for _ in 1:n_samples]
    y = [Tensor(zeros(Float64, n_classes)) for _ in 1:n_samples]
    
    for i in 1:n_samples
        class = rand(1:n_classes)
        y[i].data[class] = 1.0
    end
    
    return X, y
end

# Función auxiliar para crear modelo de prueba
function create_test_model(input_size=10, hidden_size=20, output_size=3)
    return Sequential([
        Dense(input_size, hidden_size),
        Activation(relu),
        Dense(hidden_size, hidden_size),
        Activation(relu),
        Dense(hidden_size, output_size)
    ])
end

# Función auxiliar para crear historia sintética
function create_test_history(epochs=10)
    history = History()
    history.epochs = epochs
    
    # Generar datos sintéticos de pérdida
    for i in 1:epochs
        # Pérdida decreciente con algo de ruido
        train_loss = 1.0 / (i + 1) + rand() * 0.1
        val_loss = 1.0 / (i + 0.5) + rand() * 0.15
        
        push!(history.train_loss, Float32(train_loss))
        push!(history.val_loss, Float32(val_loss))
        
        # Métricas crecientes
        train_acc = min(0.99, i / epochs + rand() * 0.1)
        val_acc = min(0.95, i / epochs + rand() * 0.1 - 0.05)
        
        if !haskey(history.train_metrics, "accuracy")
            history.train_metrics["accuracy"] = Float32[]
            history.val_metrics["accuracy"] = Float32[]
        end
        
        push!(history.train_metrics["accuracy"], Float32(train_acc))
        push!(history.val_metrics["accuracy"], Float32(val_acc))
    end
    
    return history
end

@testset "ReportGenerator Tests" begin
    
    @testset "1. ReportTemplate" begin
        println("\n1️⃣ Test de ReportTemplate")
        
        # Template por defecto
        template = ReportTemplate()
        @test template.header == "DeepDynamics Training Report"
        @test !isempty(template.css_style)
        @test !isempty(template.plot_config)
        @test :executive_summary in template.sections
        @test :model_architecture in template.sections
        @test :training_progress in template.sections
        
        # Template personalizado
        custom_template = ReportTemplate(
            header="Mi Reporte Personalizado",
            sections=[:executive_summary, :training_progress]
        )
        @test custom_template.header == "Mi Reporte Personalizado"
        @test length(custom_template.sections) == 2
        
        println("  ✅ ReportTemplate funcionando correctamente")
    end
    
    @testset "2. Generación de Reporte HTML" begin
        println("\n2️⃣ Test de generación HTML")
        
        # Crear datos de prueba
        model = create_test_model()
        history = create_test_history(15)
        config = Dict{String,Any}(
            "batch_size" => 32,
            "learning_rate" => 0.001,
            "optimizer" => "Adam",
            "epochs" => 15,
            "training_time" => 120.5,
            "random_seed" => 42
        )
        
        # Generar reporte HTML
        temp_dir = mktempdir()
        report_path = joinpath(temp_dir, "test_report")
        
        filename = generate_training_report(
            model, history, config;
            format=:html,
            save_path=report_path
        )
        
        @test isfile(filename)
        @test endswith(filename, ".html")
        
        # Verificar contenido
        content = read(filename, String)
        @test occursin("DeepDynamics Training Report", content)
        @test occursin("Executive Summary", content)
        @test occursin("Model Architecture", content)
        @test occursin("Training Progress", content)
        @test occursin("plotly", content)  # Verificar plots interactivos
        
        # Limpiar
        rm(temp_dir, recursive=true)
        
        println("  ✅ Generación HTML funcionando correctamente")
    end
    
    @testset "3. Generación de Reporte Markdown" begin
        println("\n3️⃣ Test de generación Markdown")
        
        model = create_test_model()
        history = create_test_history(10)
        config = Dict{String,Any}(
            "batch_size" => 64,
            "learning_rate" => 0.01,
            "optimizer" => "SGD",
            "epochs" => 10
        )
        
        temp_dir = mktempdir()
        report_path = joinpath(temp_dir, "test_report")
        
        filename = generate_training_report(
            model, history, config;
            format=:markdown,
            save_path=report_path
        )
        
        @test isfile(filename)
        @test endswith(filename, ".md")
        
        # Verificar contenido Markdown
        content = read(filename, String)
        @test occursin("# DeepDynamics Training Report", content)
        @test occursin("## Executive Summary", content)
        @test occursin("| Metric | Value |", content)  # Tablas
        @test occursin("```julia", content)  # Código
        
        # Limpiar
        rm(temp_dir, recursive=true)
        
        println("  ✅ Generación Markdown funcionando correctamente")
    end
    
    @testset "4. Generación de Reporte LaTeX" begin
        println("\n4️⃣ Test de generación LaTeX")
        
        model = create_test_model(5, 10, 2)
        history = create_test_history(5)
        config = Dict{String,Any}(
            "batch_size" => 16,
            "learning_rate" => 0.001,
            "optimizer" => "Adam",
            "epochs" => 5
        )
        
        temp_dir = mktempdir()
        report_path = joinpath(temp_dir, "test_report")
        
        filename = generate_training_report(
            model, history, config;
            format=:latex,
            save_path=report_path
        )
        
        @test isfile(filename)
        @test endswith(filename, ".tex")
        
        # Verificar contenido LaTeX
        content = read(filename, String)
        @test occursin("\\documentclass", content)
        @test occursin("\\begin{document}", content)
        @test occursin("\\section{Executive Summary}", content)
        @test occursin("\\end{document}", content)
        
        # Limpiar
        rm(temp_dir, recursive=true)
        
        println("  ✅ Generación LaTeX funcionando correctamente")
    end
    
    @testset "5. ReportCallback" begin
        println("\n5️⃣ Test de ReportCallback")
        
        # Crear datos sintéticos
        X, y = create_test_data(50, 8, 2)
        model = create_test_model(8, 16, 2)
        
        # Crear callback de reporte
        temp_dir = mktempdir()
        report_cb = ReportCallback(
            output_format=:html,
            save_path=joinpath(temp_dir, "training_report"),
            generate_intermediate=true,
            intermediate_frequency=3
        )
        
        # Entrenar con el callback
        history = fit!(model, X, y,
            epochs=6,
            batch_size=10,
            optimizer=Adam(0.01f0),
            callbacks=[report_cb],
            verbose=false
        )
        
        # Verificar que se generó el reporte final
        final_report = joinpath(temp_dir, "training_report.html")
        @test isfile(final_report)
        
        # Verificar reportes intermedios
        intermediate_report = joinpath(temp_dir, "training_report_epoch_3.html")
        @test isfile(intermediate_report)
        
        # Limpiar
        rm(temp_dir, recursive=true)
        
        println("  ✅ ReportCallback funcionando correctamente")
    end
    
    @testset "6. Análisis de Performance" begin
        println("\n6️⃣ Test de análisis de performance")
        
        # Crear historia con overfitting simulado
        history = History()
        history.epochs = 20
        
        # Pérdida de entrenamiento decrece, validación aumenta después de época 10
        for i in 1:20
            train_loss = 1.0 / (i + 1)
            val_loss = i <= 10 ? 1.0 / (i + 0.5) : 1.0 / 10 + (i - 10) * 0.01
            
            push!(history.train_loss, Float32(train_loss))
            push!(history.val_loss, Float32(val_loss))
        end
        
        config = Dict{String,Any}("epochs" => 20)
        
        # Recopilar datos del reporte
        report_data = ReportGenerator.collect_report_data(
            create_test_model(), 
            history, 
            config
        )
        
        # Verificar detección de overfitting
        @test report_data["executive_summary"]["potential_overfitting"] == true
        
        # Verificar mejor época
        performance = report_data["performance_analysis"]
        @test haskey(performance, "best_epoch")
        @test performance["best_epoch"] == 10  # Debería ser alrededor de época 10
        
        println("  ✅ Análisis de performance funcionando correctamente")
    end
    
    @testset "7. Hardware Utilization" begin
        println("\n7️⃣ Test de hardware utilization")
        
        config = Dict{String,Any}()
        hardware_info = ReportGenerator.collect_hardware_info()
        
        # Verificar información básica
        @test haskey(hardware_info, "cpu_threads")
        @test hardware_info["cpu_threads"] > 0
        @test haskey(hardware_info, "gpu_available")
        
        # Si hay GPU disponible, verificar info adicional
        if CUDA.functional()
            @test hardware_info["gpu_available"] == true
            @test haskey(hardware_info, "gpu_name")
            @test haskey(hardware_info, "gpu_memory_total")
            @test hardware_info["gpu_memory_total"] > 0
        else
            @test hardware_info["gpu_available"] == false
        end
        
        println("  ✅ Hardware info funcionando correctamente")
    end
    
    @testset "8. Templates Personalizables" begin
        println("\n8️⃣ Test de templates personalizables")
        
        # CSS personalizado
        custom_css = """
        <style>
            body { background: #000; color: #fff; }
        </style>
        """
        
        # Secciones personalizadas
        custom_sections = [:executive_summary, :hardware_utilization]
        
        template = ReportTemplate(
            header="Reporte Personalizado",
            css_style=custom_css,
            sections=custom_sections
        )
        
        model = create_test_model()
        history = create_test_history(5)
        config = Dict{String,Any}("epochs" => 5)
        
        temp_dir = mktempdir()
        report_path = joinpath(temp_dir, "custom_report")
        
        filename = generate_training_report(
            model, history, config;
            format=:html,
            template=template,
            save_path=report_path
        )
        
        content = read(filename, String)
        @test occursin("Reporte Personalizado", content)
        @test occursin("background: #000", content)
        @test occursin("Executive Summary", content)
        @test occursin("Hardware Utilization", content)
        @test !occursin("Model Architecture", content)  # No incluida
        
        # Limpiar
        rm(temp_dir, recursive=true)
        
        println("  ✅ Templates personalizables funcionando")
    end
    
    @testset "9. Reproducibilidad" begin
        println("\n9️⃣ Test de información de reproducibilidad")
        
        Random.seed!(12345)
        config = Dict{String,Any}(
            "random_seed" => 12345,
            "batch_size" => 32,
            "learning_rate" => 0.001,
            "optimizer" => "Adam",
            "epochs" => 10
        )
        
        repro_info = ReportGenerator.collect_reproducibility_info(config)
        
        @test repro_info["random_seed"] == 12345
        @test repro_info["batch_size"] == 32
        @test repro_info["learning_rate"] == 0.001
        @test repro_info["julia_version"] == string(VERSION)
        
        println("  ✅ Info de reproducibilidad funcionando")
    end
    
    @testset "10. Integración con Training Real" begin
        println("\n🔟 Test de integración completa")
        
        # Datos más realistas
        X, y = create_test_data(200, 15, 4)
        model = Sequential([
            Dense(15, 30),
            Activation(relu),
            Dense(30, 20),
            Activation(relu),
            Dense(20, 4),
            Activation(softmax)
        ])
        
        # Configurar reporte automático
        temp_dir = mktempdir()
        report_cb = ReportCallback(
            output_format=:html,
            save_path=joinpath(temp_dir, "full_report")
        )
        
        # Entrenar
        start_time = time()
        history = fit!(model, X, y,
            epochs=8,
            batch_size=20,
            optimizer=Adam(0.001f0),
            loss_fn=categorical_crossentropy,
            metrics=[:accuracy],
            validation_split=0.2f0,
            callbacks=[report_cb],
            verbose=false
        )
        training_time = time() - start_time
        
        # Verificar reporte generado
        report_file = joinpath(temp_dir, "full_report.html")
        @test isfile(report_file)
        
        # Verificar contenido completo
        content = read(report_file, String)
        @test occursin("DeepDynamics Training Report", content)
        @test occursin("Total Parameters", content)
        @test occursin("Best Training Loss", content)
        @test occursin("plotly", content)
        @test occursin("Epoch", content)
        
        # Verificar que incluye información real del entrenamiento
        @test occursin("accuracy", content)
        @test occursin("0.001", content)  # learning rate
        
        # Generar también versión Markdown para verificar
        md_file = generate_training_report(
            model, history, 
            Dict("training_time" => training_time);
            format=:markdown,
            save_path=joinpath(temp_dir, "full_report_md")
        )
        
        md_content = read(md_file, String)
        @test occursin("## Executive Summary", md_content)
        @test occursin("| Metric | Value |", md_content)
        
        # Limpiar
        rm(temp_dir, recursive=true)
        
        println("  ✅ Integración completa funcionando")
    end
    
    @testset "11. Manejo de Errores" begin
        println("\n1️⃣1️⃣ Test de manejo de errores")
        
        # Formato no soportado
        @test_throws AssertionError generate_training_report(
            create_test_model(),
            create_test_history(),
            Dict{String,Any}();
            format=:invalid_format
        )
        
        # Historia vacía
        empty_history = History()
        temp_dir = mktempdir()
        
        # No debería fallar, sino generar reporte vacío
        filename = generate_training_report(
            create_test_model(),
            empty_history,
            Dict{String,Any}();
            save_path=joinpath(temp_dir, "empty_report")
        )
        
        @test isfile(filename)
        
        # Limpiar
        rm(temp_dir, recursive=true)
        
        println("  ✅ Manejo de errores funcionando")
    end
    
    @testset "12. Formatos de Salida" begin
        println("\n1️⃣2️⃣ Test de todos los formatos de salida")
        
        model = create_test_model()
        history = create_test_history(5)
        config = Dict{String,Any}("epochs" => 5, "batch_size" => 32)
        
        temp_dir = mktempdir()
        
        formats = [:html, :markdown, :latex]
        extensions = [".html", ".md", ".tex"]
        
        for (fmt, ext) in zip(formats, extensions)
            filename = generate_training_report(
                model, history, config;
                format=fmt,
                save_path=joinpath(temp_dir, "report_$fmt")
            )
            
            @test isfile(filename)
            @test endswith(filename, ext)
            @test filesize(filename) > 100  # No vacío
        end
        
        # Limpiar
        rm(temp_dir, recursive=true)
        
        println("  ✅ Todos los formatos funcionando")
    end

    @testset "Generación HTML en src/reports" begin
    println("\n🔍 Test HTML en carpeta src/reports")

    # 1) Calculamos él path absoluto a src/reports
    # @__DIR__ es ...\DeepDynamics\test
    test_dir    = @__DIR__
    project_dir = normpath(joinpath(test_dir, ".."))          # ...\DeepDynamics
    src_dir     = joinpath(project_dir, "src")                # ...\DeepDynamics\src
    reports_dir = joinpath(src_dir, "reports")                # ...\DeepDynamics\src\reports

    println("► reports_dir = $reports_dir")
    mkpath(reports_dir)   # crea src/reports si no existe

    # 2) Generamos datos de prueba
    model   = create_test_model()
    history = create_test_history(5)
    config  = Dict("batch_size"=>16, "learning_rate"=>0.01, "optimizer"=>"SGD", "epochs"=>5)

    # 3) Llamamos al report generator
    filename = generate_training_report(
        model, history, config;
        format    = :html,
        save_path = joinpath(reports_dir, "html_src_test")
    )

    # 4) Aserciones
    @test isfile(filename)
    @test endswith(filename, ".html")
    content = read(filename, String)
    @test occursin("DeepDynamics Training Report", content)

    # 5) Listamos el contenido de la carpeta para depurar
    println("Contenido de reports_dir: ", readdir(reports_dir))

    # 6) Limpiar
    rm(reports_dir; recursive=true, force=true)

    println("  ✅ HTML en src/reports generado correctamente")
end


end

println("\n" * "="^60)
println("✅ Todos los tests de ReportGenerator pasaron exitosamente!")
println("="^60)