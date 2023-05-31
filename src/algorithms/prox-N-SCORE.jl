using SelfConcordantSmoothOptimization

export ProxNSCORE

# A Proximal Newton method
Base.@kwdef mutable struct ProxNSCORE <: ProximalMethod
    ss_type::Int = 1
    name::String = "prox-newtonscore"
    label::String = "Prox-N-SCORE"
end
init!(method::ProxNSCORE, x) = method
function step!(method::ProxNSCORE, reg_name, model, hμ, As, x, x_prev, ys, iter)
    gr = hμ.grad(x)
    λgr = model.λ .* gr
    Hr_diag = hμ.hess(x)
    λHr = model.λ .* Diagonal(Hr_diag)
    obj = x -> model.f(x) + get_reg(model, x, reg_name)
    if all(x->x!==nothing,(model.grad_fx, model.hess_fx))
        H = model.hess_fx(x)
        grad_f = x -> model.grad_fx(x)
    else
        f = model.f
        H = hessian(f, x)
        grad_f = x -> gradient(f, x)
    end
    ∇f = grad_f(x) + λgr
    d = -(H + λHr) \ ∇f

    if method.ss_type == 1 && model.L !==nothing
        step_size = 1/model.L
    elseif method.ss_type == 1 && model.L === nothing
        step_size = 0.5
    elseif method.ss_type == 2
        if iter == 1
            step_size = 1
        else
            λgr_prev = model.λ .* hμ.grad(x_prev)
            ∇f_prev = grad_f(x_prev) + λgr_prev
            step_size = inv_BB_step(x, x_prev, ∇f, ∇f_prev) # inverse of the original BB step-size
        end
    elseif method.ss_type == 3
        step_size = linesearch(x, d, obj, grad_f)
    else
        Base.error("Please, choose ss_type in [1, 2, 3].")
    end

    Hdiag_inv = 1 ./ Hr_diag
    H_inv = Diagonal(Hdiag_inv)

    Mg = get_Mg(hμ.Mh, hμ.ν, hμ.μ, length(x))

    d_norm = λgr' * (H_inv * λgr)
    η = sqrt(d_norm)
    α = step_size/(1 + Mg*η)

    # ensure αₖ satisfies the theoretical condition
    # (actually satisfies it for many convex problems)
    safe_α = min(1, α) 
    prox_m = invoke_prox(model, reg_name, x + safe_α*d, Hdiag_inv, model.λ, step_size)
    x_new = prox_step(prox_m)

    return x_new
end