using DeepDynamics
using DeepDynamics.ReportGenerator

# Crear carpeta reports si no existe
reports_dir = joinpath(@__DIR__, "..", "src", "reports")
mkpath(reports_dir)

# Datos de ejemplo
X, y     = create_test_data(50, 8, 2)      # o tu propio dataset
model    = create_test_model(8, 16, 2)     # tu modelo de prueba
history  = fit!(model, X, y; epochs=6, batch_size=10, verbose=false)
config   = Dict("batch_size"=>10, "learning_rate"=>0.01, "epochs"=>6)

# Generar reporte definitivo
filename = generate_training_report(
    model,
    history,
    config;
    format    = :html,
    save_path = joinpath(reports_dir, "manual_report")
)

println("📂 Reporte disponible en: $filename")
