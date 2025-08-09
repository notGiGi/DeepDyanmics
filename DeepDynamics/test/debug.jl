using DeepDynamics

# Crea un ejemplo mínimo
ln = LayerNorm(10)
X = randn(Float32, 4, 10)   # igual que en tu test

# Paso 1: calcula estadísticos
N       = prod(ln.normalized_shape)
sum_X   = sum(X, dims=2)
mean_X  = sum_X ./ Float32(N)
diff    = X .- mean_X
sum_sq  = sum(diff .^ 2, dims=2)
var_X   = sum_sq ./ Float32(N)
inv_std = 1f0 ./ sqrt.(var_X .+ ln.eps)

# Paso 2: calcula reshape_shape tal como lo haces tú
D          = ndims(X)
nd         = length(ln.normalized_shape)
norm_dims  = (D - nd + 1):D
reshape_shape = ntuple(i-> i in norm_dims ? ln.normalized_shape[i-(D-nd)] : size(X,i), D)

# Ahora copia aquí los resultados:
println("size(X)          = ", size(X))
println("size(sum_X)      = ", size(sum_X))
println("size(mean_X)     = ", size(mean_X))
println("size(var_X)      = ", size(var_X))
println("size(inv_std)    = ", size(inv_std))
println("reshape_shape    = ", reshape_shape)
