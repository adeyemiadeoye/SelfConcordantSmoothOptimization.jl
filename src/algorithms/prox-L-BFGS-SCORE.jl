using SelfConcordantSmoothOptimization

export ProxLQNSCORE

# A Proximal Limited-Memory Quasi-Newton method (L-BFGS)
Base.@kwdef mutable struct ProxLQNSCORE <: ProximalMethod
    """
    ProxLQNSCORE

    Proximal Limited-Memory Quasi-Newton (L-BFGS) method with self-concordant regularization.

    # Fields
    - `ss_type`: Step size type (1: fixed, 2: Barzilai-Borwein, 3: line search)
    - `use_prox`: Whether to use the proximal step
    - `m`: Memory size for L-BFGS
    - `s_list`: List of s vectors (x_{k+1} - x_k)
    - `y_list`: List of y vectors (grad_{k+1} - grad_k)
    - `H0`: Initial Hessian scaling (scalar)
    - `name`: Algorithm name
    - `label`: Human-readable label
    """
    ss_type::Int = 1
    use_prox::Bool = true
    m::Int = 10
    s_list::Vector{Vector{Float64}} = Vector{Vector{Float64}}()
    y_list::Vector{Vector{Float64}} = Vector{Vector{Float64}}()
    H0::Float64 = 1.0
    name::String = "prox-lbfgsscore"
    label::String = "Prox-LBFGS-SCORE"
end
function init!(method::ProxLQNSCORE, x)
    method.s_list = Vector{Vector{Float64}}()
    method.y_list = Vector{Vector{Float64}}()
    method.H0 = 1.0
    return method
end
function set_name!(method::ProxLQNSCORE, implemented_algs)
    if method.use_prox == false
        method.name = "lbfgsscore"
        method.label = "LBFGS-SCORE"
        push!(implemented_algs, method.name)
    else
        push!(implemented_algs, method.name)
    end
    return method
end
function two_loop_recursion(method::ProxLQNSCORE, grad)
    q = copy(grad)
    alpha = Float64[]
    rho = Float64[]
    for (s, y) in zip(reverse(method.s_list), reverse(method.y_list))
        rho_i = 1.0 / dot(y, s)
        alpha_i = rho_i * dot(s, q)
        q -= alpha_i * y
        push!(alpha, alpha_i)
        push!(rho, rho_i)
    end
    r = method.H0 * q
    for i in 1:length(method.s_list)
        s = method.s_list[i]
        y = method.y_list[i]
        rho_i = rho[end - i + 1]
        alpha_i = alpha[end - i + 1]
        beta = rho_i * dot(y, r)
        r += s * (alpha_i - beta)
    end
    return -r
end
function step!(method::ProxLQNSCORE, model::OptimModel, reg_name, hμ, As, x, x_prev, ys, Cmat, iter; ∇fx=nothing, return_dx=false)
    if length(model.λ) > 1
        # λ = 1.0 # pre-multiplication will done for more than one regularization function
        λ = model.λ[1]
    else
        λ = model.λ
    end
    gr = hμ.grad(Cmat,x)
    λgr = λ .* gr
    Hr_diag = hμ.hess(Cmat,x)
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
    if ∇fx !== nothing
        grad_f = x -> ∇fx
    end
    ∇q = grad_f(x) + λgr
    if iter == 1 || isempty(method.s_list)
        d = -∇q
    else
        d = two_loop_recursion(method, ∇q)
    end

    if method.ss_type == 1 && model.L !==nothing
        step_size = min(1/model.L,1.0)
    elseif method.ss_type == 1 && model.L === nothing
        step_size = 0.5
    elseif method.ss_type == 2 || model.L === nothing
        if iter == 1
            step_size = 1
        else
            λgr_prev = λ .* hμ.grad(Cmat,x_prev)
            ∇q_prev = grad_f(x_prev) + λgr_prev
            step_size = inv_BB_step(x, x_prev, ∇q, ∇q_prev) # BB step-size
        end
    elseif method.ss_type == 3
        grad_q = (x) -> grad_f(x) + λ.*hμ.grad(Cmat,x)
        step_size = linesearch(x, d, obj, grad_q)
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
    dx = safe_α*d
    if method.use_prox
        prox_m = invoke_prox(model, reg_name, x + dx, Hdiag_inv, λ, step_size)
        x_new = prox_step(prox_m)
        δh = x_new - x
    else
        x_new = x + dx
        δh = dx
    end

    gr_new = hμ.grad(Cmat,x_new)
    λgr_new = λ .* gr_new
    ∇q_new = grad_f(x_new) + λgr_new
    γh = ∇q_new - ∇q

    # uppdate memory
    if dot(δh, γh) > 1e-10
        if length(method.s_list) == method.m
            popfirst!(method.s_list)
            popfirst!(method.y_list)
        end
        push!(method.s_list, δh)
        push!(method.y_list, γh)
        method.H0 = dot(γh, δh) / dot(γh, γh)
    end
    pri_res_norm = norm(δh)
    if return_dx
        return x_new, dx, pri_res_norm
    else
        return x_new, pri_res_norm
    end
end