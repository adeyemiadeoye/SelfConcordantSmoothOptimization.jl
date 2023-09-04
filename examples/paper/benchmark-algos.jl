using Dates
using ProximalOperators
using ProximalAlgorithms

import Base.show


struct BMSolution{A, O, R, R2, T, K, M}
    x::A
    obj::O
    rel::R
    objrel::R2
    times::T
    iters::K
    model::M
end

struct BMSolutionPlus{A, O, R, R2, T, K, D, M}
    x::A
    obj::O
    rel::R
    objrel::R2
    times::T
    iters::K
    metrics::D
    model::M
end

show(io::IO, s::BMSolution) = show(io, "")
show(io::IO, s::BMSolutionPlus) = show(io, "")

Base.@kwdef mutable struct BMPANOC
    name::String = "panoc"
    label::String = "PANOC"
end
function get_iter_states(method::BMPANOC, x0, f, g, Lf)
    if Lf === nothing
        return ProximalAlgorithms.PANOCIteration(x0=x0, f=f, g=g)
    else
        return ProximalAlgorithms.PANOCIteration(x0=x0, f=f, g=g, Lf=Lf)
    end
end

Base.@kwdef mutable struct BMZeroFPR
    name::String = "zerofpr"
    label::String = "ZeroFPR"
end
function get_iter_states(method::BMZeroFPR, x0, f, g, Lf)
    if Lf === nothing
        return ProximalAlgorithms.ZeroFPRIteration(x0=x0, f=f, g=g)
    else
        return ProximalAlgorithms.ZeroFPRIteration(x0=x0, f=f, g=g, Lf=Lf)
    end
end

Base.@kwdef mutable struct BMFastProxGrad
    name::String = "f-prox-grad"
    label::String = "Fast Prox-Grad"
end
function get_iter_states(method::BMFastProxGrad, x0, f, g, Lf)
    if Lf === nothing
        return ProximalAlgorithms.FastForwardBackwardIteration(x0=x0, f=f, g=g)
    else
        return ProximalAlgorithms.FastForwardBackwardIteration(x0=x0, f=f, g=g, Lf=Lf)
    end
end

function iterate_bmark(model, iter_states, reg_name; x_tol=1e-10, f_tol=1e-10, extra_metrics=false)
    if extra_metrics
        deconv_metrics = ["psnr", "mse", "re", "se", "sparsity_level"]
        metrics = Dict(metric=>[] for metric in deconv_metrics)
    end
    objs = []
    rel_errors = []
    f_rel_errors = []
    times = []
    x = model.x0
    x_star = copy(model.x)
    obj_star = model.f(x_star) + get_reg(model, x_star, reg_name)
    n_iter = 0
    t0 = now()
    for state in iter_states
        Δtime = (now() - t0).value/1000
        n_iter += 1
        rel_error = max(norm(x - x_star) / max.(norm(x_star),1), x_tol)
        obj = model.f(x) + get_reg(model, x, reg_name)
        f_rel_error = max((norm(obj - obj_star))/norm(obj_star), f_tol)
        push!(times, Δtime)
        println("Iter $n_iter \t Loss: $obj \t x_rel: $rel_error \t Time: $Δtime")
        flush(stdout)
        push!(objs, obj)
        push!(rel_errors, rel_error)
        push!(f_rel_errors, f_rel_error)
        
        # if we are interested in metrics related to a sparse deconvolution problem
        if extra_metrics
            push!(metrics["psnr"], psnr_metric(model.x, x))
            push!(metrics["mse"], mean_square_error(model.x, x))
            push!(metrics["re"], recon_error(model.x, x))
            push!(metrics["se"], support_error(model.x, x; eps=1e-3))
            push!(metrics["sparsity_level"], sparsity_level(x))
        end
        
        x_new = copy(state.x)
        if n_iter > 1 && norm(x_new - x) / (1 + norm(x_new)) <= x_tol
            break
        end
        x = x_new
    end

    n_iter = length(rel_errors)

    if extra_metrics
        return BMSolutionPlus(x, objs, rel_errors, f_rel_errors, times, n_iter, metrics, model)
    else
        return BMSolution(x, objs, rel_errors, f_rel_errors, times, n_iter, model)
    end
end