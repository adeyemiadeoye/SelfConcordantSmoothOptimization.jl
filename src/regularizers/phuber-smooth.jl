# pseudo-Huber smoothing function

const huber_smooth_Mh = 2.0
const huber_smooth_ν = 2.6

mutable struct PHuberSmootherL1L2 <: Smoother
    """
    PHuberSmootherL1L2

    Smoother for pseudo-Huber approximation of the l1/l2 regularizer.

    # Fields
    - `μ`: Smoothing parameter
    - `Mh`: Smoothness constant
    - `ν`: Self-concordance parameter
    - `val`: Value function (smoothed regularizer)
    - `grad`: Gradient function
    - `hess`: Hessian function
    """
    μ
    Mh
    ν
    val
    grad
    hess
end
PHuberSmootherL1L2(mu::IntOrFloat; val=x->pseudo_huber.(x;μ=mu), grad=(Cmat,x)->huber_grad.(x;μ=mu), hess=(Cmat,x)->huber_hess.(x;μ=mu)) = PHuberSmootherL1L2(mu, huber_smooth_Mh, huber_smooth_ν, val, grad, hess)
function pseudo_huber(x; μ=1.0, λ=1.0)
    return (μ^2 - μ*sqrt(μ^2 + x^2) + x^2)*(μ^2 + x^2)^(-1/2)
end
function huber_grad(x; μ=1.0, λ=1.0)
    x * (μ^2 + x^2)^-(1/2)
end
function huber_hess(x; μ=1.0, λ=1.0) # returns a vector, the diagonal part of huber_hess (a diagonal matrix)
    μ^2 * (μ^2 + x^2)^-(3/2)
end

mutable struct PHuberSmootherIndBox <: Smoother
    """
    PHuberSmootherIndBox

    Smoother for pseudo-Huber approximation of the indicator box regularizer.

    # Fields
    - `μ`: Self-concordance/Smoothing parameter
    - `Mh`: Self-concordance constant
    - `ν`: Self-concordance parameter
    - `val`: Value function (smoothed regularizer)
    - `grad`: Gradient function
    - `hess`: Hessian function
    """
    μ
    Mh
    ν
    val
    grad
    hess
end
function PHuberSmootherIndBox(lb::VectorOrFloat, ub::VectorOrFloat, mu::IntOrFloat)
    val = x -> pseudo_huber_indbox(x;μ=mu,lb=lb,ub=ub)
    grad = (Cmat,x) -> huber_grad_indbox(x;μ=mu,lb=lb,ub=ub)
    hess = (Cmat,x) -> huber_hess_indbox(x;μ=mu,lb=lb,ub=ub)
    
    return PHuberSmootherIndBox(mu, huber_smooth_Mh, huber_smooth_ν, val, grad, hess)
end
function pseudo_huber_indbox(x; μ=1.0,lb=0.0,ub=1.0)
    n = size(x,1)
    h = Vector{Float64}(undef, n)

    a, b = bounds_sanity_check(n, lb, ub)

    for i in 1:n
        if x[i] < a[i]
            h[i] = (-μ*sqrt(a[i]^2 - 2*a[i]*x[i] + μ^2 + x[i]^2) + μ^2 + (-x[i] + a[i])^2) * (a[i]^2 - 2*a[i]*x[i] + μ^2 + x[i]^2)^(-1/2)
        elseif x[i] == a[i] || x[i] <= b[i]
            h[i] = eps()
        else
            h[i] = (-μ*sqrt(b[i]^2 - 2*b[i]*x[i] + μ^2 + x[i]^2) + μ^2 + (b[i] - x[i])^2)*(b[i]^2 - 2*b[i]*x[i] + μ^2 + x[i]^2)^(-1/2)
        end
    end
    return h
end
function huber_grad_indbox(x; μ=1.0,lb=0.0,ub=1.0)
    n = size(x,1)
    g = Vector{Float64}(undef, n)

    a, b = bounds_sanity_check(n, lb, ub)
    for i in 1:n
        if -x[i] < a[i]
            g[i] = (a[i]^2 - 2*x[i]*a[i] + μ^2 + x[i]^2)^(-1/2)*(-x[i] + a[i])
        elseif x[i] == a[i] || x[i] < b[i]
            g[i] = eps()
        else
            g[i] = (b[i]^2 - 2*b[i]*x[i] + μ^2 + x[i]^2)^(-1/2)*(b[i] - x[i])
        end
    end
    return g
end
function huber_hess_indbox(x; μ=1.0,lb=0.0,ub=1.0) # returns a vector, the diagonal part of huber_hess (a diagonal matrix)
    n = size(x,1)
    h = Vector{Float64}(undef, n)

    a, b = bounds_sanity_check(n, lb, ub)
    for i in 1:n
        if x[i] <= a[i]
            h[i] = μ^2*(a[i]^2 - 2*a[i]*x[i] + μ^2 + x[i]^2)^(-3/2)
        elseif a[i] < x[i] < b[i]
            h[i] = eps()
        elseif x[i] >= b[i]
            h[i] = μ^2*(b[i]^2 - 2*b[i]*x[i] + μ^2 + x[i]^2)^(-3/2)
        end
    end
    return h
end

mutable struct PHuberSmootherGL <: Smoother
    """
    PHuberSmootherGL

    Smoother for pseudo-Huber approximation of the group lasso regularizer.

    # Fields
    - `μ`: Smoothing parameter
    - `Mh`: Smoothness constant
    - `ν`: Self-concordance parameter
    - `val`: Value function (smoothed regularizer)
    - `grad`: Gradient function
    - `hess`: Hessian function
    """
    μ
    Mh
    ν
    val
    grad
    hess
end
function PHuberSmootherGL(mu::IntOrFloat, model)
    λ = model.λ
    P = model.P
    inds = P.ind
    grpNUM = P.grpNUM
    λ1, λ2 = λ[1], λ[2]
    
    val = x -> get_infconvHuberL2L1(x, λ1, λ2, inds, grpNUM, mu)
    grad = (Cmat,x) -> huber_l2l1_grad(Cmat, x, λ1, λ2, inds, grpNUM, mu)
    hess = (Cmat,x) -> huber_l2l1_hess(Cmat, x, λ1, λ2, inds, grpNUM, mu)
    return PHuberSmootherGL(mu, huber_smooth_Mh, huber_smooth_ν, val, grad, hess)
end

function huber_l2l1_grad(Cmat, x, λ1, λ2, inds, grpNUM, μ)
    g_mu1 = pseudo_huber.(x;μ=μ,λ=1)
    Dg_mu1 = huber_grad.(x;μ=μ,λ=1)

    return huber_grad.(Cmat*g_mu1;μ=μ,λ=1).*Dg_mu1
end
function huber_l2l1_hess(Cmat, x, λ1, λ2, inds, grpNUM, μ)
    g_mu1 = pseudo_huber.(x;μ=μ,λ=1)
    Dg_mu1 = huber_grad.(x;μ=μ,λ=1)
    DDg_mu1 = huber_hess.(x;μ=μ,λ=1)

    DDgg = huber_hess.(Cmat*g_mu1;μ=μ,λ=1)*dot(Dg_mu1,Dg_mu1) .+ huber_grad.(Cmat*g_mu1;μ=μ,λ=1).*DDg_mu1

    return DDgg
end

function infconvHuberNorm(x::Vector{Float64}, λ::IntOrFloat, inds::Matrix{Int}, grpNUM::Int, μ::IntOrFloat)
    ICz = similar(x)

    ind = reduce(vcat, inds)

    for j in 1:grpNUM
        λw = λ * ind[2+3*(j-1)+1]
        kstart = Int(ind[3*(j-1)+1])
        kend = Int(ind[1+3*(j-1)+1])
        nrm = twonorm(x, kstart, kend)

        for k in kstart:kend
            ICz[k] = x[k] * max(1 - λw / nrm, 0)
            ICz[k] = pseudo_huber(ICz[k]; μ=μ, λ=λw)
        end
    end

    return ICz
end

function get_infconvHuberL2L1(x, λ1, λ2, inds, grpNUM, μ)
    utmp = infconvHuberNorm(x, λ1, inds, grpNUM, μ)
    z = infconvHuberNorm(utmp, λ2, inds, grpNUM, μ)
    return z
end