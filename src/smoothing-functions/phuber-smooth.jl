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
function PHuberSmootherIndBox(lb::IntOrFloat, ub::IntOrFloat, mu::IntOrFloat)
    val = x -> pseudo_huber_indbox.(x;μ=mu,a=lb,b=ub)
    grad = x -> huber_grad_indbox.(x;μ=mu,a=lb,b=ub)
    hess = x -> huber_hess_indbox.(x;μ=mu,a=lb,b=ub)
    
    return PHuberSmootherIndBox(mu, exp_smooth_Mh, exp_smooth_ν, val, grad, hess)
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

function pseudo_huber_indbox(x; μ=1.0,a=0.0,b=1.0)
    if x < a
        return (-μ*sqrt(a^2 - 2*a*x + μ^2 + x^2) + μ^2 + (-x + a)^2) * (a^2 - 2*a*x + μ^2 + x^2)^(-1/2)
    elseif x == a || x <= b
        return 1e-9
    else
        return (-μ*sqrt(b^2 - 2*b*x + μ^2 + x^2) + μ^2 + (b - x)^2)*(b^2 - 2*b*x + μ^2 + x^2)^(-1/2)
    end
end
function huber_grad_indbox(x; μ=1.0,a=0.0,b=1.0)
    if -x < a
        return (a^2 - 2*x*a + μ^2 + x^2)^(-1/2)*(-x + a)
    elseif x == a || x < b
        return 1e-9
    else
        (b^2 - 2*b*x + μ^2 + x^2)^(-1/2)*(b - x)
    end
end
function huber_hess_indbox(x; μ=1.0,a=0.0,b=1.0) # returns a vector, the diagonal part of huber_hess (a diagonal matrix)
    if x <= a
        return μ^2*(a^2 - 2*a*x + μ^2 + x^2)^(-3/2)
    elseif a < x < b
        return 1e-9
    elseif x >= b
        return μ^2*(b^2 - 2*b*x + μ^2 + x^2)^(-3/2)
    end
end