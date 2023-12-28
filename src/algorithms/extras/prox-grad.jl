using SelfConcordantSmoothOptimization

export ProxGradient

# Proximal gradient method
Base.@kwdef mutable struct ProxGradient <: ProximalMethod
    ss_type::Int = 1
    use_prox::Bool = true
    name::String = "prox-grad"
    label::String = "Prox-Grad"
end
init!(method::ProxGradient, x) = method
function step!(method::ProxGradient, reg_name, model, hμ, As, x, x_prev, ys, Cmat, iter)
    if length(model.λ) > 1
        λ = 1.0 # pre-multiplication will done for more than one regularization function
    else
        λ = model.λ
    end
    λgr = λ .* hμ.grad(Cmat,x)
    Hr_diag = hμ.hess(Cmat,x)
    obj = x -> model.f(As, ys, x) + get_reg(model, x, reg_name)
    if model.grad_fx !== nothing
        grad_f = x -> model.grad_fx(x)
    else
        f = x -> model.f(As, ys, x)
        grad_f = x -> gradient(f, x)
    end
    ∇f = grad_f(x) + λgr
    d = -∇f

    if method.ss_type == 1 && model.L !==nothing
        step_size = 1/model.L
    elseif method.ss_type == 1 && model.L === nothing
        step_size = linesearch(x, d, obj, grad_f)
    elseif method.ss_type == 2
        λgr_prev = λ .* hμ.grad(x_prev)
        ∇f_prev = grad_f(x_prev) + λgr_prev
        δ = x - x_prev
        γ = ∇f - ∇f_prev

        L_val = (γ ⋅ γ)/(δ' * γ) # Barzilai-Borwein (BB) step-size
        # L_val = (δ' if iter == 1
        if iter == 1
            step_size = 1
        else
            step_size = L_val # inverse of the original BB step-size
        end
    elseif method.ss_type == 3
        step_size = linesearch(x, d, obj, grad_f)
    else
        step_size = 1/model.L
    end

    if method.use_prox
        prox_m = invoke_prox(model, reg_name, x + step_size*d, one.(Hr_diag), λ, step_size)
        x_new = prox_step(prox_m)
    else
        x_new = x + step_size*d
    end

    return x_new
end