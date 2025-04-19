# Exponential smoothing function for both lower and upper bounds

mutable struct LogExpSmootherIndBox <: Smoother
    """
    LogExpSmootherIndBox

    Smoother for log-exponential approximation of the indicator box regularizer.

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

const LogExp_smooth2_Mh = 1.0
const LogExp_smooth2_ν = 2.0

function LogExpSmootherIndBox(lb::VectorOrFloat, ub::VectorOrFloat, mu::IntOrFloat)
    val = x -> LogExp_smooth_indbox(x; μ=mu, lb=lb, ub=ub)
    grad = (Cmat, x) -> LogExp_smooth_grad_indbox(x; μ=mu, lb=lb, ub=ub)
    hess = (Cmat, x) -> LogExp_smooth_hess_indbox(x; μ=mu, lb=lb, ub=ub)
    
    return LogExpSmootherIndBox(mu, LogExp_smooth2_Mh, LogExp_smooth2_ν, val, grad, hess)
end

function LogExp_smooth_indbox(x; μ=1.0, lb=0.0, ub=1.0)
    n = size(x,1)
    a, b = bounds_sanity_check(n, lb, ub)
    return @. ifelse(x <= a + μ, (a - x + 3*μ)*(a - x + μ)/(2*μ),
           ifelse(x >= b - μ, (x - b + 3*μ)*(x - b + μ)/(2*μ),
           0.0)) + @. ifelse(x < a, μ * (log(μ) - log(x - a)),
           ifelse(x > b, μ * (log(μ) - log(b - x)), 0.0))
end

function LogExp_smooth_grad_indbox(x; μ=1.0, lb=0.0, ub=1.0)
    n = size(x,1)
    a, b = bounds_sanity_check(n, lb, ub)
    return @. ifelse(x <= a + μ, (x - a - 2*μ)/μ,
           ifelse(x >= b - μ, (x - b + 2*μ)/μ,
           0.0)) + @. ifelse(x < a, μ/(a - x),
           ifelse(x > b, -μ/(b - x), 0.0))
end

function LogExp_smooth_hess_indbox(x; μ=1.0, lb=0.0, ub=1.0)
    n = size(x,1)
    a, b = bounds_sanity_check(n, lb, ub)
    return @. ifelse(x <= a + μ, 1/μ,
           ifelse(x >= b - μ, 1/μ,
           0.0)) + @. ifelse(x < a, μ/(a - x)^2,
           ifelse(x > b, μ/(b - x)^2, 0.0))
end
