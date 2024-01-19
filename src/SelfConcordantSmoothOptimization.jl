module SelfConcordantSmoothOptimization

export ProximalMethod
export Solution
export SolutionPlus
export iterate!
export NoSmooth
export PHuberSmootherL1L2, PHuberSmootherFL, PHuberSmootherGL, PHuberSmootherIndBox
export OsBaSmootherL1L2, OsBaSmootherGL
export ExponentialSmootherIndBox
export get_reg

using LinearAlgebra
using ForwardDiff: gradient, hessian, jacobian
using Dates

import Base.show

include("problems.jl")
include("prox-operators.jl")
include("utils/alg-utils.jl")

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

function iter_step!(method::ProximalMethod, model::ProxModel, reg_name, hμ, As, x, x_prev, ys, Cmat, iter)
    return Vector{Float64}(step!(method, model, reg_name, hμ, As, x, x_prev, ys, Cmat, iter))
end

# for each implemented algorithm, add the name field here
# TODO automate this in an efficient way
implemented_algs = ["prox-newtonscore", "prox-ggnscore", "prox-bfgsscore"]
function iterate!(method::ProximalMethod, model::ProxModel, reg_name, hμ; α=nothing, batch_size=nothing, max_iter=1000, x_tol=1e-10, f_tol=1e-10, extra_metrics=false, verbose=1)
    m = size(model.y,1)
    n = size(model.x,1)
    if method.name in implemented_algs
        if α !== nothing
            model.L = 1/α
        end
        if method.ss_type == 1 && model.L === nothing && verbose > 0
            @info "Neither L nor α is set for the problem... Now fixing α = 0.5..."
        end
    end
    if batch_size !== nothing && batch_size < m
        data = batch_data(model)
        minibatch_mode = true
    else
        minibatch_mode = false
        batch_size = m
    end
    if extra_metrics
        deconv_metrics = ["psnr", "mse", "re", "se", "sparsity_level"]
        metrics = Dict(metric=>[] for metric in deconv_metrics)
    end
    objs = []
    rel_errors = []
    f_rel_errors = []
    times = []
    iters = 0
    x_star = model.x
    f(x) = model.f(model.A, model.y, x)
    obj_star = f(x_star) + get_reg(model, x_star, reg_name)
    x = model.x0
    x_prev = deepcopy(x)
    init!(method, x)
    t0 = now()
    for iter in 1:max_iter
        Δtime = (now() - t0).value/1000

        obj = f(x) + get_reg(model, x, reg_name)

        if reg_name == "gl"# && model.L !==nothing
            Cmat = model.P.Cmat
            rel_error = mean_square_error(x_star, x)
        else
            Cmat = I
            rel_error = max(norm(x - x_star) / max.(norm(x_star),1), x_tol)
        end
        
        f_rel_error = max((norm(obj - obj_star))/norm(obj_star), f_tol)
        push!(times, Δtime)
        if verbose > 1
            solve_info = "Epoch $(iter-1) \t Loss: $obj \t x_rel: $rel_error \t Time: $Δtime"
            println(solve_info)
            flush(stdout)
        end
        push!(objs, obj)
        push!(rel_errors, rel_error)
        push!(f_rel_errors, f_rel_error)

        # if we are interested in metrics related to some a sparse deconvolution problem
        if extra_metrics
            push!(metrics["psnr"], psnr_metric(model.x, x))
            push!(metrics["mse"], mean_square_error(model.x, x))
            push!(metrics["re"], recon_error(model.x, x))
            push!(metrics["se"], support_error(model.x, x; eps=1e-3))
            push!(metrics["sparsity_level"], sparsity_level(x))
        end

        for i in 1:batch_size:m
            if minibatch_mode
                As, ys = sample_batch(data, batch_size)
            else
                As, ys = model.A, model.y
            end

            if verbose > 2
                print("[$(Int(ceil(i/batch_size)))/$(Int(ceil(m/batch_size)))] ")
                if iter==max_iter && i==(1:batch_size:m)[end]
                    Δtime = (now() - t0).value/1000
                    obj = f(x) + get_reg(model, x, reg_name)
                    solve_info = "\nEpoch $iter \t Loss: $obj \t x_rel: $rel_error \t Time: $Δtime"
                    println(solve_info)
                    flush(stdout)
                end
            end

            x_new = iter_step!(method, model, reg_name, hμ, As, x, x_prev, ys, Cmat, iter)
            if norm(x_new - x) < x_tol*max.(norm(x), 1) || f_rel_error ≤ f_tol
                break
            end
            x_prev = deepcopy(x)
            x = x_new
        end

        if norm(x - x_prev) < x_tol*max.(norm(x_prev), 1) || f_rel_error ≤ f_tol
            break
        end

        iters += 1
    end
    if extra_metrics
        return SolutionPlus(x, objs, rel_errors, f_rel_errors, times, iters, metrics, model)
    else
        return Solution(x, objs, rel_errors, f_rel_errors, times, iters, model)
    end
end

function batch_data(model::ProxModel)
    m = size(model.y, 1)
    return [(model.A[i,:],model.y[i,:]) for i in 1:m]
end

function sample_batch(data, mb)
    # mb : batch_size
    s = sample(data,mb,replace=false,ordered=true)
    As = Array(hcat(map(x->x[1], s)...)')
    ys = Array(hcat(map(x->x[2], s)...)')

    return As, ys
end

include("algorithms/prox-N-SCORE.jl")
include("algorithms/prox-GGN-SCORE.jl")
include("algorithms/prox-BFGS-SCORE.jl")
# extras
include("algorithms/extras/prox-grad.jl")
include("algorithms/extras/OWLQN.jl")

include("smoothing-functions/smoothing.jl")
include("smoothing-functions/phuber-smooth.jl")
include("smoothing-functions/ostrovskii-bach-smooth.jl")
include("smoothing-functions/exponential-smooth.jl")


export mean_square_error, psnr_metric, recon_error, sparsity_level, support_error

end