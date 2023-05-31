# Logistic smoothing function

mutable struct LogisticSmootherL1 <: Smoother
    μ
    Mh
    ν
    val
	grad
    hess
end

const log_smooth_Mh = 1.0
const log_smooth_ν = 2.0

LogisticSmootherL1(mu::IntOrFloat; val=x->log_smooth_l1.(x;μ=mu), grad=x->log_smooth_grad_l1.(x;μ=mu), hess=x->log_smooth_hess_l1.(x;μ=mu)) = LogisticSmootherL1(mu, log_smooth_Mh, log_smooth_ν, val, grad, hess)

function log_smooth_l1(x; μ=1.0)
    return μ*log(1 + exp(x/μ))
end
function log_smooth_grad_l1(x; μ=1.0)
    exp(x/μ) / (1 + exp(x/μ))
end
function log_smooth_hess_l1(x; μ=1.0) # returns a vector, the diagonal part of log_smooth_hess_l1 (a diagonal matrix)
    exp(x/μ) / (μ*(1 + exp(x/μ))^2)
end