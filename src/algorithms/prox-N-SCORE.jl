using SelfConcordantSmoothOptimization

export ProxNSCORE

# A Proximal Newton method
Base.@kwdef mutable struct ProxNSCORE <: ProximalMethod
    ss_type::Int = 1
    use_prox::Bool = true
    name::String = "prox-newtonscore"
    label::String = "Prox-N-SCORE"
end
init!(method::ProxNSCORE, x) = method
function set_name!(method::ProxNSCORE, implemented_algs)
    if method.use_prox == false
        method.name = "newtonscore"
        method.label = "Newton-SCORE"
        push!(implemented_algs, "newtonscore")
    else
        push!(implemented_algs, method.name)
    end
    return method
end
function step!(method::ProxNSCORE, model::ProxModel, reg_name, hμ, As, x, x_prev, ys, Cmat, iter)
    if length(model.λ) > 1
        λ = model.λ[1]
    else
        λ = model.λ
    end
    gr = hμ.grad(Cmat,x)
    λgr = λ .* gr
    Hr_diag = hμ.hess(Cmat,x)
    λHr = λ .* Diagonal(Hr_diag)
    obj = x -> model.f(As, ys, x) + get_reg(model, x, reg_name)
    if all(x->x!==nothing,(model.grad_fx, model.hess_fx))
        H = model.hess_fx(As, ys, x)
        grad_f = x -> model.grad_fx(As, ys, x)
    else
        f = x -> model.f(As, ys, x)
        H = hessian(f, x)
        grad_f = x -> gradient(f, x)
    end
    ∇f = grad_f(x) + λgr
    d = -(H + λHr) \ ∇f

    if method.ss_type == 1 && model.L !==nothing
        step_size = min(1/model.L,1.0)
    elseif method.ss_type == 1 && model.L === nothing
        step_size = 0.5
    elseif method.ss_type == 2
        if iter == 1
            step_size = 1
        else
            λgr_prev = λ .* hμ.grad(x_prev)
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
    if method.use_prox
        prox_m = invoke_prox(model, reg_name, x + safe_α*d, Hdiag_inv, λ, step_size)
        x_new = prox_step(prox_m)
    else
        x_new = x + safe_α*d
    end

    return x_new
end