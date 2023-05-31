using LambertW

# Exponential smoothing function

mutable struct ExponentialSmootherL1 <: Smoother
    μ
    Mh
    ν
    val
    grad
    hess
end

mutable struct ExponentialSmootherL2 <: Smoother
    μ
    Mh
    ν
    val
    grad
    hess
end

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

ExponentialSmootherL1(mu::IntOrFloat; val=x->exp_smooth_l1.(x;μ=mu), grad=x->exp_smooth_grad_l1.(x;μ=mu), hess=x->exp_smooth_hess_l1.(x;μ=mu)) = ExponentialSmootherL1(mu, exp_smooth_Mh, exp_smooth_ν, val, grad, hess)

ExponentialSmootherL2(mu::IntOrFloat; val=x->exp_smooth_l2.(x;μ=mu), grad=x->exp_smooth_grad_l2.(x;μ=mu), hess=x->exp_smooth_hess_l2.(x;μ=mu)) = ExponentialSmootherL2(mu, exp_smooth_Mh, exp_smooth_ν, val, grad, hess)

function ExponentialSmootherIndBox(lb::IntOrFloat, ub::IntOrFloat, mu::IntOrFloat)
    val = x -> exp_smooth_indbox.(x;μ=mu,a=lb,b=ub)
    grad = x -> exp_smooth_grad_indbox.(x;μ=mu,a=lb,b=ub)
    hess = x -> exp_smooth_hess_indbox.(x;μ=mu,a=lb,b=ub)
    
    return ExponentialSmootherIndBox(mu, exp_smooth_Mh, exp_smooth_ν, val, grad, hess)
end

function exp_smooth_l1(x; μ=1.0)
    x >= 0 ? μ * exp.(-x/μ) : μ - x
end
function exp_smooth_grad_l1(x; μ=1.0)
    x < 0 ? -1.0 : x >= 0 ? exp(-x/μ) : 0.0
end
function exp_smooth_hess_l1(x; μ=1.0) # returns a vector, the diagonal part of exp_smooth_hess_l1 (a diagonal matrix)
    x < 0 ? 0.0 : 1/μ * exp(-x/μ)
end

function exp_smooth_l2(x; μ=1.0)
    μ^2 * lambertw(1/μ*exp(-x/μ))*(lambertw(1/μ * exp(-x/μ)) + 2)/2
end
function exp_smooth_grad_l2(x; μ=1.0)
    -μ*lambertw(1/μ*exp(-x/μ))
end
function exp_smooth_hess_l2(x; μ=1.0) # returns a vector, the diagonal part of exp_smooth_hess_l2 (a diagonal matrix)
    lambertw(1/μ*exp(-x/μ))/(1+lambertw(1/μ*exp(-x/μ)))
end

function exp_smooth_indbox(x; μ=1.0, a=0.0, b=1.0)
    exp((-x + a) / μ) * μ
end
function exp_smooth_grad_indbox(x; μ=1.0, a=0.0, b=1.0)
    -exp((-x + a) / μ)
end
function exp_smooth_hess_indbox(x; μ=1.0, a=0.0, b=1.0) # returns a vector, the diagonal part of exp_smooth_hess_l2 (a diagonal matrix)
    1/μ * exp((-x + a)/μ)
end