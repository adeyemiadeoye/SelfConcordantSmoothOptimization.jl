# Exponential smoothing function

mutable struct ExponentialSmootherIndBox <: Smoother
    μ
    Mh
    ν
    val
    grad
    hess
end

const exp_smooth_Mh = 1.0
const exp_smooth_ν = 2.0

function ExponentialSmootherIndBox(lb::VectorOrFloat, ub::VectorOrFloat, mu::IntOrFloat)
    val = x -> exp_smooth_indbox(x;μ=mu,lb=lb,ub=ub)
    grad = (Cmat,x) -> exp_smooth_grad_indbox(x;μ=mu,lb=lb,ub=ub)
    hess = (Cmat,x) -> exp_smooth_hess_indbox(x;μ=mu,lb=lb,ub=ub)
    
    return ExponentialSmootherIndBox(mu, exp_smooth_Mh, exp_smooth_ν, val, grad, hess)
end

function exp_smooth_indbox(x; μ=1.0, lb=0.0, ub=1.0)
    n = size(x,1)
    a, b = bounds_sanity_check(n, lb, ub)
    return @. exp((-x + a) / μ) * μ
end
function exp_smooth_grad_indbox(x; μ=1.0, lb=0.0, ub=1.0)
    n = size(x,1)
    a, b = bounds_sanity_check(n, lb, ub)
    return @. -exp((-x + a) / μ)
end
function exp_smooth_hess_indbox(x; μ=1.0, lb=0.0, ub=1.0) # returns a vector, the diagonal part of exp_smooth_hess_l2 (a diagonal matrix)
    n = size(x,1)
    a, b = bounds_sanity_check(n, lb, ub)
    return @. 1/μ * exp((-x + a)/μ)
end