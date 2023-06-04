# pseudo-Huber smoothing function

mutable struct PHuberSmootherL1L2 <: Smoother
    μ
    Mh
    ν
    val
    grad
    hess
end

mutable struct PHuberSmootherIndBox <: Smoother
    μ
    Mh
    ν
    val
    grad
    hess
end

const huber_smooth_Mh = 2.0
const huber_smooth_ν = 2.6

PHuberSmootherL1L2(mu::IntOrFloat; val=x->pseudo_huber.(x;μ=mu), grad=x->huber_grad.(x;μ=mu), hess=x->huber_hess.(x;μ=mu)) = PHuberSmootherL1L2(mu, huber_smooth_Mh, huber_smooth_ν, val, grad, hess)
function PHuberSmootherIndBox(lb::VectorOrFloat, ub::VectorOrFloat, mu::IntOrFloat)
    val = x -> pseudo_huber_indbox(x;μ=mu,lb=lb,ub=ub)
    grad = x -> huber_grad_indbox(x;μ=mu,lb=lb,ub=ub)
    hess = x -> huber_hess_indbox(x;μ=mu,lb=lb,ub=ub)
    
    return PHuberSmootherIndBox(mu, huber_smooth_Mh, huber_smooth_ν, val, grad, hess)
end

function pseudo_huber(x; μ=1.0)
    return (μ^2 - μ*sqrt(μ^2 + x^2) + x^2)*(μ^2 + x^2)^(-1/2)
end
function huber_grad(x; μ=1.0)
    x * ((μ^2 + x^2)^-(1/2))
end
function huber_hess(x; μ=1.0) # returns a vector, the diagonal part of huber_hess (a diagonal matrix)
    μ^2 * ((μ^2 + x^2)^-(3/2))
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