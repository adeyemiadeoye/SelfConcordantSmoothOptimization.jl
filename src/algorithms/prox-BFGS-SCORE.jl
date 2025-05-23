using SelfConcordantSmoothOptimization

export ProxQNSCORE

# A Proximal Quasi-Newton method (BFGS)
Base.@kwdef mutable struct ProxQNSCORE <: ProximalMethod
    """
    ProxQNSCORE

    Proximal Quasi-Newton (BFGS) method with self-concordant regularization.

    # Fields
    - `ss_type`: Step size type (1: fixed, 2: Barzilai-Borwein, 3: line search)
    - `use_prox`: Whether to use the proximal step
    - `H`: Hessian approximation matrix
    - `name`: Algorithm name
    - `label`: Human-readable label
    """
    ss_type::Int = 1
    use_prox::Bool = true
    H::Matrix{Float64} = Matrix{Float64}(I, 1, 1)
    name::String = "prox-bfgsscore"
    label::String = "Prox-QN-SCORE (BFGS)"
end
function init!(method::ProxQNSCORE, x)
	method.H = Matrix{Float64}(I, size(x,1), size(x,1))
	return method
end
function set_name!(method::ProxQNSCORE, implemented_algs)
    if method.use_prox == false
        method.name = "bfgsscore"
        method.label = "BFGS-SCORE"
        push!(implemented_algs, method.name)
    else
        push!(implemented_algs, method.name)
    end
    return method
end
function step!(method::ProxQNSCORE, model::OptimModel, reg_name, hμ, As, x, x_prev, ys, Cmat, iter)
    if length(model.λ) > 1
        # λ = 1.0 # pre-multiplication will done for more than one regularization function
        λ = model.λ[1]
    else
        λ = model.λ
    end
    H = method.H
    gr = hμ.grad(Cmat,x)
    λgr = λ .* gr
    Hr_diag = hμ.hess(Cmat,x)
    λHr = λ .* Diagonal(Hr_diag)
    if typeof(model) <: ModelGeneric
        obj = x -> model.f(x) + get_reg(model, x, reg_name)
    else
        obj = x -> model.f(As, ys, x) + get_reg(model, x, reg_name)
    end
    if model.grad_fx !== nothing
        if typeof(model) <: ModelGeneric
            grad_f = x -> model.grad_fx(x)
        else
            grad_f = x -> model.grad_fx(As, ys, x)
        end
    else
        if typeof(model) <: ModelGeneric
            f = x -> model.f(x)
        else
            f = x -> model.f(As, ys, x)
        end
        grad_f = x -> gradient(f, x)
    end
    ∇f = grad_f(x) + λgr
    d = -(H + λHr)\∇f

    if method.ss_type == 1 && model.L !==nothing
        step_size = min(1/model.L,1.0)
    elseif method.ss_type == 1 && model.L === nothing
        step_size = 0.5
    elseif method.ss_type == 2 || model.L === nothing
        if iter == 1
            step_size = 1
        else
            λgr_prev = λ .* hμ.grad(x_prev)
            ∇f_prev = grad_f(x_prev) + λgr_prev
            step_size = inv_BB_step(x, x_prev, ∇f, ∇f_prev) # BB step-size
        end
    elseif method.ss_type == 3
        step_size = linesearch(x, d, obj, grad_f)
    else
        Base.error("Please, choose ss_type in [1, 2, 3].")
    end
    
    Hdiag_inv = 1 ./ Hr_diag
    H_inv = Diagonal(Hdiag_inv)

    Mg = get_Mg(hμ.Mh, hμ.ν, hμ.μ, length(x))

    η = sqrt(λgr' * (H_inv * λgr))
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

    ∇f_new = grad_f(x_new)
    δh = x_new - x
    γh = ∇f_new - ∇f

    H[:] = H - (δh*γh'*H + H*γh*δh')/(δh'*γh) + (1 + (γh'*H*γh)/(δh'*γh))[1]*(δh*δh')/(δh'*γh)
    return x_new
end