# Burg smoothing function

mutable struct BurgSmootherL1 <: Smoother
    μ
    Mh
    ν
    val
	grad
    hess
end

mutable struct BurgSmootherL2 <: Smoother
    μ
    Mh
    ν
    val
	grad
    hess
end

const burg_smooth_Mh = 8.0
const burg_smooth_ν = 3.0

BurgSmootherL1(mu::IntOrFloat; val=x->burg_smooth_l1.(x;μ=mu), grad=x->burg_smooth_grad_l1.(x;μ=mu), hess=x->burg_smooth_hess_l1.(x;μ=mu)) = BurgSmootherL1(mu, burg_smooth_Mh, burg_smooth_ν, val, grad, hess)

BurgSmootherL2(mu::IntOrFloat; val=x->burg_smooth_l2.(x;μ=mu), grad=x->burg_smooth_grad_l2.(x;μ=mu), hess=x->burg_smooth_hess_l2.(x;μ=mu)) = BurgSmootherL2(mu, burg_smooth_Mh, burg_smooth_ν, val, grad, hess)

function burg_smooth_l1(x; μ=1.0)
    x < μ/2 ? μ*log(2) + μ - 2*x : μ*(log(μ) - log(x))/2
end
function burg_smooth_grad_l1(x; μ=1.0)
    x < μ/2 ? -1.0 : -μ/(2*x)
end
function burg_smooth_hess_l1(x; μ=1.0) # returns a vector, the diagonal part of burg_smooth_hess_l1 (a diagonal matrix)
    x < μ/2 ? 1e-9 : μ/(2*x^2)
end

function burg_smooth_l2(x; μ=1.0)
    x^2/4 - x*sqrt(x^2 + 2*μ)/4 + μ/4 + μ*log(-x + sqrt(x^2 + 2*μ))/2
end
function burg_smooth_grad_l2(x; μ=1.0)
    -(-x*sqrt(x^ 2 + 2*μ) + x^2 + 2*μ)*(x^2 + 2*μ)^(-1/2)/2
end
function burg_smooth_hess_l2(x; μ=1.0) # returns a vector, the diagonal part of burg_smooth_hess_l2 (a diagonal matrix)
    (-x + sqrt(x^2 + 2*μ))*(x^2 + 2*μ)^(-1/2)/2
end