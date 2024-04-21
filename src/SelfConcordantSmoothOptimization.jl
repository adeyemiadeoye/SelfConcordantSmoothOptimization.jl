module SelfConcordantSmoothOptimization

export ProximalMethod
export Solution
export iterate!
export NoSmooth
export PHuberSmootherL1L2, PHuberSmootherFL, PHuberSmootherGL, PHuberSmootherIndBox
export OsBaSmootherL1L2, OsBaSmootherGL
export ExponentialSmootherIndBox
export get_reg

using LinearAlgebra
using MLUtils
using ForwardDiff: gradient, hessian, jacobian
using Dates

import Base.show

include("problems.jl")
include("prox-operators.jl")
include("utils/alg-utils.jl")
include("utils/utils.jl")

abstract type ProximalMethod end

struct Solution{A, O, R, R2, D, T, K, M}
    x::A
    obj::O
    fval::O
    fvaltest::O
    rel::R
    objrel::R2
    metricvals::D
    times::T
    epochs::K
    model::M
end

show(io::IO, s::Solution) = show(io, "")
show(io::IO, p::Problem) = show(io, "")

function iter_step!(method::ProximalMethod, model::ProxModel, reg_name, hμ, As, x, x_prev, ys, Cmat, iter)
    return Vector{Float64}(step!(method, model, reg_name, hμ, As, x, x_prev, ys, Cmat, iter))
end

function iterate!(method::ProximalMethod, model::ProxModel, reg_name, hμ; metrics::Union{Dict{A,B},Nothing}=nothing, α=nothing, batch_size=nothing, slice_samples=false, shuffle_batch=true, max_epoch=1000, x_tol=1e-10, f_tol=1e-10, verbose=1) where {A,B}
    implemented_algs = []
    set_name!(method, implemented_algs)
    m = size(model.y,1)
    ny = size(model.y,2)
    if method.name in implemented_algs
        if α !== nothing
            model.L = 1/α
        end
        if method.ss_type == 1 && model.L === nothing && verbose > 0
            @info "Neither L nor α is set for the problem... Now fixing α = 0.5..."
        end
    end
    if batch_size !== nothing && slice_samples
        @info "Cannot use both batch_size and slice_samples=true...\nNow setting slice_samples=false..."
        slice_samples = false # prioritize mini-batching
    end
    if slice_samples
        data = slice_data(model)
        batch_size = 1
    elseif batch_size !== nothing
        data = DataLoader((data=model.A', label=model.y'), batchsize=batch_size, shuffle=shuffle_batch)
    else
        batch_size = m
        data = DataLoader((data=model.A', label=model.y'), batchsize=batch_size)
    end
    fvals = []
    fvaltests = []
    objs = []
    rel_errors = []
    f_rel_errors = []
    metric_vals = Dict()
    if metrics !== nothing
        for name in keys(metrics)
            metric_vals[name] = []
        end
    end
    times = []
    epochs = 0
    x_star = model.x
    f(x) = model.f(model.A, model.y, x)
    test_model = all(x->x!==nothing, (model.Atest, model.ytest))
    if xor(model.Atest===nothing, model.ytest===nothing)
        @info "Both input (Atest) and target (ytest) data are required for testing the model, but only one of these has been provided.\nWill skip testing..."
    elseif test_model
        ftest(x) = model.f(model.Atest, model.ytest, x)
    end
    obj_star = f(x_star) + get_reg(model, x_star, reg_name)
    x = model.x0
    x_prev = deepcopy(x)
    l_split = "="^30*"\n"
    init!(method, x)
    t0 = now()
    for epoch_t in 1:max_epoch
        Δtime = (now() - t0).value/1000
        epoch = epoch_t-1

        fval = f(x)
        obj = fval + get_reg(model, x, reg_name)
        if test_model
            fvaltest = ftest(x)
        end

        if reg_name == "gl"# && model.L !==nothing
            Cmat = model.P.Cmat
            rel_error = mean_square_error(x_star, x)
        else
            Cmat = I
            rel_error = max(norm(x - x_star) / max.(norm(x_star),1), x_tol)
        end
        
        f_rel_error = max((norm(obj - obj_star))/norm(obj_star), f_tol)
        push!(times, Δtime)
        if metrics !== nothing
            for name in keys(metrics)
                push!(metric_vals[name], metrics[name](model, x))
            end
        end
        if verbose > 1
            @eval(@showval("Optimizer", $method.label))
            if test_model
                push!(fvaltests, fvaltest)
                @show epoch obj fval fvaltest rel_error Δtime
            else
                @show epoch obj fval rel_error Δtime
            end
            if metrics !== nothing
                for name in keys(metrics)
                    @eval(@showval($name, $metric_vals[$name][$epoch_t]))
                end
            end
            print(l_split)
        end
        push!(objs, obj)
        push!(fvals, fval)
        push!(rel_errors, rel_error)
        push!(f_rel_errors, f_rel_error)

        iend = Int(ceil(m/batch_size))
        for (i, sample) in enumerate(data)
            As, ys = sample
            ny == 1 ? (As, ys) = (Matrix(As'), vec(ys')) : (As, ys) = (Matrix(As'), Matrix(ys'))

            if verbose > 2
                if (i in [1, iend]) || (i%100==0)
                    print("\n[$i/$iend]")
                else
                    print("#")
                end
                if epoch_t==max_epoch && i==iend
                    Δtime = (now() - t0).value/1000
                    fval = f(x)
                    obj = fval + get_reg(model, x, reg_name)
                    print("\n"*l_split)
                    @eval(@showval("Optimizer", $method.label))
                    if test_model
                        fvaltest = ftest(x)
                        push!(fvaltests, fvaltest)
                        @show max_epoch obj fval fvaltest rel_error Δtime
                    else
                        @show max_epoch obj fval rel_error Δtime
                    end
                    if metrics !== nothing
                        for name in keys(metrics)
                            push!(metric_vals[name], metrics[name](model, x))
                        end
                        for name in keys(metrics)
                            @eval(@showval($name, $metric_vals[$name][end]))
                        end
                    end
                    f_rel_error = max((norm(obj - obj_star))/norm(obj_star), f_tol)
                    push!(objs, obj)
                    push!(fvals, fval)
                    push!(rel_errors, rel_error)
                    push!(f_rel_errors, f_rel_error)
                    push!(times, Δtime)
                end
            end

            x_new = iter_step!(method, model, reg_name, hμ, As, x, x_prev, ys, Cmat, epoch_t)
            if norm(x_new - x) < x_tol*max.(norm(x), 1) || f_rel_error ≤ f_tol
                if epoch_t != max_epoch
                    terminate_epoch = epoch_t
                    Δtime = (now() - t0).value/1000
                    fval = f(x)
                    obj = fval + get_reg(model, x, reg_name)
                    if verbose > 2
                        print("\n"*l_split)
                        @eval(@showval("Optimizer", $method.label))
                    end
                    if test_model
                        fvaltest = ftest(x)
                        push!(fvaltests, fvaltest)
                        if verbose > 2
                            @show terminate_epoch obj fval fvaltest rel_error Δtime
                        end
                    elseif verbose > 2
                        @show terminate_epoch obj fval rel_error Δtime
                    end
                    if metrics !== nothing
                        for name in keys(metrics)
                            push!(metric_vals[name], metrics[name](model, x))
                        end
                        if verbose > 2
                            for name in keys(metrics)
                                @eval(@showval($name, $metric_vals[$name][end]))
                            end
                        end
                    end
                    f_rel_error = max((norm(obj - obj_star))/norm(obj_star), f_tol)
                    push!(objs, obj)
                    push!(fvals, fval)
                    push!(rel_errors, rel_error)
                    push!(f_rel_errors, f_rel_error)
                    push!(times, Δtime)
                end
                break
            end
            x_prev = deepcopy(x)
            x = x_new
        end

        if norm(x - x_prev) < x_tol*max.(norm(x_prev), 1) || f_rel_error ≤ f_tol
            break
        end

        epochs += 1
        if verbose > 2
            print("\n"*l_split)
        end
    end
    return Solution(x, objs, fvals, fvaltests, rel_errors, f_rel_errors, metric_vals, times, epochs, model)
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


export mean_square_error

end