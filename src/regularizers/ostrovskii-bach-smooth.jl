# Ostrovskii & Bach smoothing function

const osba_smooth_Mh = 2*sqrt(2)
const osba_smooth_ν = 3.0

mutable struct OsBaSmootherL1L2 <: Smoother
    """
    OsBaSmootherL1L2

    Smoother for Ostrovskii & Bach approximation of the l1/l2 regularizer.

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
OsBaSmootherL1L2(mu::IntOrFloat; val=x->osba_smooth_l1.(x;μ=mu), grad=x->osba_smooth_grad_l1.(x;μ=mu), hess=x->osba_smooth_hess_l1.(x;μ=mu)) = OsBaSmootherL1L2(mu, osba_smooth_Mh, osba_smooth_ν, val, grad, hess)
function osba_smooth_l1(x; μ=1.0, λ=1.0)
    return λ*(sqrt(μ^2 + 4*x^2)/2 - μ/2 + μ*log((2*x - sqrt(μ^2 + 4*x^2) + μ)/x)/2 - log(2)*μ + μ*log((sqrt(μ^2 + 4*x^2) - μ + 2*x)/x)/2)
end
function osba_smooth_grad_l1(x; μ=1.0, λ=1.0)
    return λ*((-μ^3 + μ^2 * sqrt(μ^2 + 4*x^2) - 4*x^2*μ + 2*x^2*sqrt(μ^2 + 4*x^2))*(μ*sqrt(μ^2 + 4*x^2) + μ^2 + 4*x^2)/(4*μ^2*x^3 + 16*x^5))
end
function osba_smooth_hess_l1(x; μ=1.0, λ=1.0) # returns a vector, the diagonal part of osba_smooth_hess (a diagonal matrix)
    return λ*((sqrt(μ^2 + 4*x^2) - μ)*μ/x^2*(μ^2 + 4*x^2)^(-1//2)/2)
end

mutable struct OsBaSmootherGL <: Smoother
    """
    OsBaSmootherGL

    Smoother for Ostrovskii & Bach approximation of the group lasso regularizer.

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
function OsBaSmootherGL(mu::IntOrFloat, model)
    λ = model.λ
    P = model.P
    inds = P.ind
    grpNUM = P.grpNUM
    λ1, λ2 = λ[1], λ[2]
    
    val = x -> get_infconvOsBaL2L1(x, λ1, λ2, inds, grpNUM, mu)
    grad = (Cmat,x) -> osba_l2l1_grad(Cmat, x, λ1, λ2, inds, grpNUM, mu)
    hess = (Cmat,x) -> osba_l2l1_hess(Cmat, x, λ1, λ2, inds, grpNUM, mu)
    return OsBaSmootherGL(mu, osba_smooth_Mh, osba_smooth_ν, val, grad, hess)
end


function osba_l2l1_grad(Cmat, x, λ1, λ2, inds, grpNUM, μ)
    g_mu1 = osba_smooth_l1.(x;μ=μ,λ=1)
    Dg_mu1 = osba_smooth_grad_l1.(x;μ=μ,λ=1)

    return osba_smooth_grad_l1.(Cmat*g_mu1;μ=μ,λ=1).*Dg_mu1
end
function osba_l2l1_hess(Cmat, x, λ1, λ2, inds, grpNUM, μ)
    g_mu1 = osba_smooth_l1.(x;μ=μ,λ=1)
    Dg_mu1 = osba_smooth_grad_l1.(x;μ=μ,λ=1)
    DDg_mu1 = osba_smooth_hess_l1.(x;μ=μ,λ=1)

    DDgg = osba_smooth_hess_l1.(Cmat*g_mu1;μ=μ,λ=1)*dot(Dg_mu1,Dg_mu1) .+ osba_smooth_grad_l1.(Cmat*g_mu1;μ=μ,λ=1).*DDg_mu1

    return DDgg
end

function infconvOsBaNorm(x::Vector{Float64}, λ::IntOrFloat, inds::Matrix{Int}, grpNUM::Int, μ::IntOrFloat)
    ICz = similar(x)

    ind = reduce(vcat, inds)

    for j in 1:grpNUM
        λw = λ * ind[2+3*(j-1)+1]
        kstart = Int(ind[3*(j-1)+1])
        kend = Int(ind[1+3*(j-1)+1])

        for k in kstart:kend
            ICz[k] = osba_smooth_l1(x[k]; μ=μ, λ=λw)
        end
    end

    return ICz
end

function get_infconvOsBaL2L1(x, λ1, λ2, inds, grpNUM, μ)
    utmp = infconvOsBaNorm(x, λ1, inds, grpNUM, μ)
    z = infconvOsBaNorm(utmp, λ2, inds, grpNUM, μ)
    return z
end