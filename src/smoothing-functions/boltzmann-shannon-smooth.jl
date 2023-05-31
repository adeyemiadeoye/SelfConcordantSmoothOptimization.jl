# Boltzmann-Shannon entropy smoothing function

mutable struct BoShSmootherL1 <: Smoother
    μ
    Mh
    ν
    val
	grad
    hess
end

const bosh_smooth_Mh = 1.0
const bosh_smooth_ν = 4.0

BoShSmootherL1(mu::IntOrFloat; val=x->bosh_smooth_l1.(x;μ=mu), grad=x->bosh_smooth_grad_l1.(x;μ=mu), hess=x->bosh_smooth_hess_l1.(x;μ=mu)) = BoShSmootherL1(mu, bosh_smooth_Mh, bosh_smooth_ν, val, grad, hess)

function bosh_smooth_l1(x; μ=1.0)
    if x < μ*exp(-1)
        return -x - exp(-1) * μ
    elseif x < μ*exp(1) || x == μ*exp(-1)
        return x*(log(x) - log(μ) - 1)
    else
        x - μ*exp(1)
    end
end
function bosh_smooth_grad_l1(x; μ=1.0)
    if x < μ*exp(-1)
        return -1.0
    elseif x == μ*exp(-1) || x < μ*exp(1)
        return log(x) - log(μ)
    else
        return 1.0
    end
end
function bosh_smooth_hess_l1(x; μ=1.0) # returns a vector, the diagonal part of bosh_smooth_hess_l1 (a diagonal matrix)
    if x < exp(-1)*μ
        1e-9
    elseif x == μ*exp(-1) || x < μ*exp(1)
        return 1/x
    else
        return 1e-9
    end
end