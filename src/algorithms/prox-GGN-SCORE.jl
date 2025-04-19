using SelfConcordantSmoothOptimization

export ProxGGNSCORE

# A Proximal GGN method
Base.@kwdef mutable struct ProxGGNSCORE <: ProximalMethod
    """
    ProxGGNSCORE

    Proximal Generalized Gauss-Newton (GGN) method with self-concordant regularization.

    # Fields
    - `ss_type`: Step size type (1: fixed, 2: Barzilai-Borwein, 3: line search)
    - `use_prox`: Whether to use the proximal step
    - `name`: Algorithm name
    - `label`: Human-readable label
    """
    ss_type::Int = 1
    use_prox::Bool = true
    name::String = "prox-ggnscore"
    label::String = "Prox-GGN-SCORE"
end
init!(method::ProxGGNSCORE, x) = method
function set_name!(method::ProxGGNSCORE, implemented_algs)
    if method.use_prox == false
        method.name = "ggnscore"
        method.label = "GGN-SCORE"
        push!(implemented_algs, method.name)
    else
        push!(implemented_algs, method.name)
    end
    return method
end
function step!(method::ProxGGNSCORE, model::OptimModel, reg_name, hμ, As, x, x_prev, ys, Cmat, iter)
    obj = x -> model.f(As, ys, x) + get_reg(model, x, reg_name)
    if length(model.λ) > 1
        λ = model.λ[1]
    else
        λ = model.λ
    end
    gr = hμ.grad(Cmat,x)
    λgr = λ .* gr
    Hr_diag = hμ.hess(Cmat,x)
    if all(x->x!==nothing,(model.jac_yx, model.grad_fy, model.hess_fy))
        ŷ = model.out_fn(As, x)
        J = model.jac_yx(As, ys, ŷ, x)
        residual = model.grad_fy(As, ys, ŷ)
        Q = model.hess_fy(As, ys, ŷ)
    else
        m_out_fn = x -> model.out_fn(As, x)
        ŷ = m_out_fn(x)
        f = ŷ -> model.f(ys, ŷ)
        J = jacobian(m_out_fn, x)
        residual = gradient(f, ŷ)
        Q = hessian(f, ŷ)
    end

    if model.grad_fx !== nothing
        grad_f = x -> model.grad_fx(x)
    else
        f = x -> model.f(As, ys, x)
        grad_f = x -> gradient(f, x)
    end

    Hdiag_inv = 1 ./ Hr_diag
    H_inv = Diagonal(Hdiag_inv)

    d = ggn_score_step(J, Q, [gr], Hr_diag, H_inv, residual, λ)

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
            step_size = inv_BB_step(x, x_prev, ∇f, ∇f_prev) # BB step-size
        end
    elseif method.ss_type == 3
        step_size = linesearch(x, d, obj, grad_f)
    else
        Base.error("Please, choose ss_type in [1, 2, 3].")
    end

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

    return x_new
end

function ggn_score_step(J::Union{SparseMatrixCSC{Float64, Int64},Matrix{Float64}}, Q::Union{Matrix{Float64}, Diagonal{Float64, Vector{Float64}}}, gr::Vector{Vector{Float64}}, Hr_diag::Vector{Float64}, H_inv::Diagonal{Float64,Vector{Float64}}, residual::VectorBitVectorOrArray2{Float64}, λ::IntOrFloat)
    n = length(gr[1])

    # Jacobian "concatenations"
    ncat = length(gr)
    qdm1 = size(Q,1)
    qdm11 = qdm1+ncat
    Jt = reduce(hcat, [J', λ .* reduce(hcat,gr)])
    residual = [vec(residual) ; ones(ncat)]
    Q = [[Q ; repeat(zeros(qdm1)', ncat)] repeat(zeros(qdm11)',ncat)']
    if qdm11 ≤ n
        A = Q * (Jt' * H_inv) * Jt
        B = qr(sum(ones(ncat))I + A) \ residual
        d = H_inv * Jt * B
    else
        JQJ = (Jt * Q * Jt') + λ.*Diagonal(Hr_diag)
        Je = Jt * residual
        d = qr(JQJ) \ Je
    end
    
    return -d
end