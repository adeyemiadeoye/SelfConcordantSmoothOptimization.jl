using SelfConcordantSmoothOptimization

export ProxGGNSCORE

# A Proximal GGN method
Base.@kwdef mutable struct ProxGGNSCORE <: ProximalMethod
    ss_type::Int = 1
    name::String = "prox-ggnscore"
    label::String = "Prox-GGN-SCORE"
end
init!(method::ProxGGNSCORE, x) = method
function step!(method::ProxGGNSCORE, reg_name, model, hμ, As, x, x_prev, ys, iter)
    obj = x -> model.f(x) + get_reg(model, x, reg_name)
    gr = hμ.grad(x)
    λgr = model.λ .* gr
    Hr_diag = hμ.hess(x)
    if all(x->x!==nothing,(model.jac_yx, model.grad_fy, model.hess_fy))
        ŷ = model.out_fn(As, x)
        J = model.jac_yx(ŷ)
        residual = model.grad_fy(ŷ)
        Q = model.hess_fy(ŷ)
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
        grad_f = x -> gradient(model.f, x)
    end

    Hdiag_inv = 1 ./ Hr_diag
    H_inv = Diagonal(Hdiag_inv)

    d = ggn_score_step(J, Q, gr, Hr_diag, H_inv, residual, model.λ, size(ys,2))

    if method.ss_type == 1 && model.L !==nothing
        step_size = min(1/model.L,1.0)
    elseif method.ss_type == 1 && model.L === nothing
        step_size = 0.5
    elseif method.ss_type == 2 || model.L === nothing
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

    Mg = get_Mg(hμ.Mh, hμ.ν, hμ.μ, length(x))

    η = sqrt(λgr' * (H_inv * λgr))
    α = step_size/(1 + Mg*η)

    # ensure αₖ satisfies the theoretical condition
    # (actually satisfies it for many convex problems)
    safe_α = min(1, α)
    
    prox_m = invoke_prox(model, reg_name, x + safe_α*d, Hdiag_inv, model.λ, step_size)
    x_new = prox_step(prox_m)

    return x_new
end

function ggn_score_step(J::Matrix{Float64}, Q::Union{Matrix{Float64}, Diagonal{Float64, Vector{Float64}}}, gr::Vector{Float64}, Hr_diag::Vector{Float64}, H_inv::Diagonal{Float64,Vector{Float64}}, residual::Array{Float64,1}, λ::Float64, ydm2::Int64)
    n = length(gr)
    qdm1 = size(Q,1)
    qdm11 = qdm1+1
    Jt = reduce(hcat, [J', λ .* gr])
    residual = [residual ; ones(ydm2)]
    Q = [[Q ; zeros(qdm1)'] zeros(qdm11)]
    if qdm11 ≤ n
        A = Q * Jt' * H_inv * Jt
        B = lu(λ.*I + A) \ residual
        d = H_inv * Jt * B
    else
        JQJ = (Jt * Q * Jt') + λ.*Diagonal(Hr_diag)
        Je = Jt * residual
        d = lu(JQJ) \ Je
    end
    
    return -d
end