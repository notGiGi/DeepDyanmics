module LRSchedulers

export StepScheduler, CosineAnnealingScheduler, get_lr

abstract type LRScheduler end

"""
Scheduler con decaimiento por pasos
"""
struct StepScheduler <: LRScheduler
    initial_lr::Float64
    step_size::Int
    gamma::Float64
end

function get_lr(scheduler::StepScheduler, epoch::Int)
    factor = scheduler.gamma ^ (epoch ÷ scheduler.step_size)
    return scheduler.initial_lr * factor
end

"""
Scheduler con annealing coseno
"""
struct CosineAnnealingScheduler <: LRScheduler
    initial_lr::Float64
    T_max::Int
    eta_min::Float64
    warmup_epochs::Int
end

function CosineAnnealingScheduler(initial_lr, T_max; eta_min=0.0, warmup_epochs=0)
    return CosineAnnealingScheduler(initial_lr, T_max, eta_min, warmup_epochs)
end

function get_lr(scheduler::CosineAnnealingScheduler, epoch::Int)
    # Fase de calentamiento (warmup)
    if epoch <= scheduler.warmup_epochs
        return scheduler.initial_lr * (epoch / max(1, scheduler.warmup_epochs))
    end
    
    # Fase de annealing
    epoch_adjusted = epoch - scheduler.warmup_epochs
    if epoch_adjusted >= scheduler.T_max
        return scheduler.eta_min
    end
    
    cos_factor = 0.5 * (1 + cos(π * epoch_adjusted / scheduler.T_max))
    return scheduler.eta_min + (scheduler.initial_lr - scheduler.eta_min) * cos_factor
end

end # module