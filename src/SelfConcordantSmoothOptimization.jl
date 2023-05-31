module SelfConcordantSmoothOptimization

export ProximalMethod
export Solution
export SolutionPlus
export iterate!
export NoSmooth
export PHuberSmootherL1L2, PHuberSmootherIndBox
export BurgSmootherL1, BurgSmootherL2
export ExponentialSmootherL1, ExponentialSmootherL2, ExponentialSmootherIndBox
export LogisticSmootherL1
export BoShSmootherL1
export get_reg

using LinearAlgebra
using ForwardDiff: gradient, hessian, jacobian
using Dates

import Base.show

include("problems.jl")
include("prox-operators.jl")
include("alg-utils.jl")

abstract type ProximalMethod end

struct Solution{A, O, R, R2, T, K, M}
    x::A
    obj::O
    rel::R
    objrel::R2
    times::T
    iters::K
    model::M
end

struct SolutionPlus{A, O, R, R2, T, K, D, M}
    x::A
    obj::O
    rel::R
    objrel::R2
    times::T
    iters::K
    metrics::D
    model::M
end

show(io::IO, s::Solution) = show(io, "")
show(io::IO, s::SolutionPlus) = show(io, "")
show(io::IO, p::Problem) = show(io, "")

function iter_step!(method::ProximalMethod, reg_name, model, hμ, As, x, x_prev, ys, iter)
    return Vector{Float64}(step!(method, reg_name, model, hμ, As, x, x_prev, ys, iter))
end

function iterate!(method::ProximalMethod, model, reg_name, hμ; α=nothing, batch_size=nothing, max_iter=1000, x_tol=1e-10, f_tol=1e-10, extra_metrics=false, verbose=0)
    N = size(model.y,1)
    if α !== nothing
        model.L = 1/α
    end
    if method.ss_type == 1 && model.L === nothing
        @info "Neither L nor α is set for the problem... Now fixing α = 0.5..."
    end
    if batch_size !== nothing && batch_size < N
        data = batch_data(model)
        minibatch_mode = true
    else
        minibatch_mode = false
    end
    if extra_metrics
        deconv_metrics = ["snr", "psnr", "mse", "re", "se", "sparsity_level"]
        metrics = Dict(metric=>[] for metric in deconv_metrics)
    end
    objs = []
    rel_errors = []
    f_rel_errors = []
    times = []
    iters = 0
    x_star = model.x
    obj_star = model.f(x_star) + get_reg(model, x_star, reg_name)
    x = model.x0
    x_prev = deepcopy(x)
    init!(method, x)
    t0 = now()
    for iter in 1:max_iter
        Δtime = (now() - t0).value/1000

        # TODO mini-batch mode not fully implemented yet
        if minibatch_mode
            As, ys = sample_batch(model, data, batch_size)
        else
            As, ys = model.A, model.y
        end
        obj = model.f(x) + get_reg(model, x, reg_name)
        rel_error = max(norm(x - x_star) / max.(norm(x_star),1), x_tol)
        f_rel_error = max((norm(obj - obj_star))/norm(obj_star), f_tol)
        push!(times, Δtime)
        if verbose > 0
            println("Iter $iter \t Loss: $obj \t Time: $Δtime")
            flush(stdout)
        end
        push!(objs, obj)
        push!(rel_errors, rel_error)
        push!(f_rel_errors, f_rel_error)

        # if we are interested in metrics related to a sparse deconvolution problem
        if extra_metrics
            push!(metrics["snr"], snr_metric(model.x, x))
            push!(metrics["psnr"], psnr_metric(model.x, x))
            push!(metrics["mse"], mean_square_error(model.x, x))
            push!(metrics["re"], recon_error(model.x, x))
            push!(metrics["se"], support_error(model.x, x; eps=1e-3))
            push!(metrics["sparsity_level"], sparsity_level(x))
        end

        x_new = iter_step!(method, reg_name, model, hμ, As, x, x_prev, ys, iter)
        if norm(x_new - x) < x_tol*max.(norm(x), 1) || f_rel_error ≤ f_tol
            break
        end
        x_prev = deepcopy(x)
        x = x_new
        iters += 1
    end
    if extra_metrics
        return SolutionPlus(x, objs, rel_errors, f_rel_errors, times, iters, metrics, model)
    else
        return Solution(x, objs, rel_errors, f_rel_errors, times, iters, model)
    end
end

include("algorithms/prox-N-SCORE.jl")
include("algorithms/prox-GGN-SCORE.jl")
include("algorithms/prox-grad.jl")
include("algorithms/prox-BFGS-SCORE.jl")
include("algorithms/OWL-QN.jl")

include("smoothing-functions/smoothing.jl")
include("smoothing-functions/phuber-smooth.jl")
include("smoothing-functions/boltzmann-shannon-smooth.jl")
include("smoothing-functions/burg-smooth.jl")
include("smoothing-functions/exponential-smooth.jl")
include("smoothing-functions/logistic-smooth.jl")


export snr_metric, mean_square_error, psnr_metric, recon_error, sparsity_level, support_error

end
