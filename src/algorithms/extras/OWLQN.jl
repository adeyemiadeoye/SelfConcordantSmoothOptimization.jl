using SelfConcordantSmoothOptimization

export OWLQN

# Orthant-Wise Limited-memory Quasi-Newton method (OWL-QN), basically an L-BFGS method
# Source and credits: the implementation here is entirely based on the GitHub Gist https://gist.github.com/yegortk/ce18975200e7dffd1759125972cd54f4
# adapted here for the ProximalMethod type (strictly for comparison and not part of the present project)
#
## References:
# Andrew, G., & Gao, J. (2007, June). Scalable training of l 1-regularized log-linear models. In Proceedings of the 24th international conference on Machine learning (pp. 33-40).
# Gong, P., & Ye, J. (2015, June). A modified orthant-wise limited memory quasi-Newton method with convergence analysis. In International Conference on Machine Learning (pp. 276-284). PMLR.

Base.@kwdef mutable struct OWLQN <: ProximalMethod
    s::Vector{Any} = []                 # param_t+1 - param_t [max size of  memory m]
    y::Vector{Any} = []                 # grad_t+1 - grad_t [max size of memory m]
    rho::Vector{Any} = []               # 1/s[i]'y[i]
    ss_type::Int = 1
    use_prox::Bool = true
    m::Int = 6           # L-BFGS history length 
    name::String = "prox-owlqn"
    label::String = "OWL-QN (L-BFGS)"
end
function init!(method::OWLQN, x)
    method.s = []
    method.y = []
    method.rho = []
    return method
end
function step!(method::OWLQN, model::ProxModel, reg_name, hμ, As, x, x_prev, ys, Cmat, iter)
    f = x -> model.f(As, ys, x)
    if length(model.λ) > 1
        # λ = 1.0 # pre-multiplication will done for more than one regularization function
        λ = model.λ[1]
    else
        λ = model.λ
    end

    reg_fn = x -> get_reg(model, x, reg_name)

    if model.grad_fx !== nothing
        ∇f = x -> model.grad_fx(x)
    else
        ∇f = x -> gradient(f, x)
    end

    s, y, rho, g = method.s, method.y, method.rho, ∇f(x)

    if all(g .== 0.0)
        println("(Local) minimum found: ∇f(x) == 0")
        x_new = x
        return x_new
    end
    
    m = min(method.m, size(s,1))
    
    x_copy = deepcopy(x)
    g_copy = deepcopy(g)
    
    pg = pseudo_gradient(g, x, λ)
    Q = deepcopy(pg)
    
    if m > 0

        # L-BFGS computation of Hessian-scaled gradient d = H_inv * g
        α = []
        for i in m : -1 : 1
            push!(α, rho[i] * (s[i] ⋅ Q))
            Q -= α[end] * y[i]
        end
        reverse!(α)
        H_inv = (s[end] ⋅ y[end]) / (y[end] ⋅ y[end])
        d = Q .* H_inv
        for i in 1:m
            d += s[i] * (α[i] - rho[i] * (y[i] ⋅ d))
        end
        
        # zeroing out all elements in d if sign(d[i]) != sign(g[i])
        # that is, if scaling changes gradient sign
        project!(d, pg)

        if method.ss_type == 1 && model.L !== nothing
            step_size = 1/model.L
        else
            step_size, x_new = projected_backtracking_line_search_update(f, reg_fn, pg, x, d)
        end
    else
        # fancy way to do  x .-= g
        d = pg
        if method.ss_type == 1 && model.L !== nothing
            step_size = 1/model.L
        else
            step_size, x_new = projected_backtracking_line_search_update(f, reg_fn, pg, x, d; α = 1 / sqrt(pg ⋅ pg), β=0.1)
        end
    end

    if method.use_prox
        prox_m = invoke_prox(model, reg_name, x - step_size*d, one.(x), λ, step_size)
        x_new = prox_step(prox_m)
    else
        x_new = x - step_size*d
    end
    
    push!(s, x - x_copy)
    push!(y, ∇f(x) - g_copy)
    push!(rho, 1/(y[end] ⋅ s[end]))
    
    while length(s) > method.m
        popfirst!(s)
        popfirst!(y)
        popfirst!(rho)
    end

    return x_new
end