module MemoryPool

export MemoryPool, get_tensor, release_tensor, preallocate_pool!

struct MemoryPool
    pool::Vector{Any}
end

function MemoryPool()
    return MemoryPool(Vector{Any}())
end

function preallocate_pool!(pool::MemoryPool, dims::NTuple{N,Int}, count::Int) where {N}
    for _ in 1:count
        push!(pool.pool, zeros(Float32, dims))
    end
end

function get_tensor(pool::MemoryPool, dims::NTuple{N,Int}) where {N}
    for (i,t) in enumerate(pool.pool)
        if size(t) == dims
            deleteat!(pool.pool, i)
            return t
        end
    end
    return zeros(Float32, dims)
end

function release_tensor(pool::MemoryPool, t)
    push!(pool.pool, t)
end

end  # module MemoryPool
