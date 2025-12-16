using StatsBase, MLUtils

function mean_square_error(xtrue, xpred)
    return mean(abs2, xtrue - xpred)
end

macro showval(name, expression)
    quote
        value = $expression
        println($name, ":\t", value)
    end
end

function slice_data(features, targets)
    return zip(eachslice(features; dims=1), eachrow(targets))
end

function get_data_loader(data; batchsize=64, shuffle=false)
    DataLoader((data=data.features', label=data.targets'), batchsize=batchsize, shuffle=shuffle)
end
function get_loader_subset(loader, indices)
    batches = collect(loader)
    subset = batches[indices]
    return subset
end

function linesearch(x, d, f, grad_f)
    α = 1.0
    rho = 0.5
    c = 1e-4
    while f(x + α*d) > f(x) + c*α*dot(grad_f(x), d)
        α = rho*α
    end
    return α
end

function update_mu(k, factor)
    return k^factor
end

# compute inverse of the Barzilai-Borwein step size to estimate alpha in the paper
## see https://en.wikipedia.org/wiki/Barzilai-Borwein_method
function inv_BB_step(x, x_prev, gradx, gradx_prev)
    δ = x - x_prev
    γ = gradx - gradx_prev
    L_est = (γ ⋅ γ)/(δ' * γ)
    return L_est
end

function show_stat!(opt, model, method, x, test_model, ftest, fvaltests, epoch, obj, fval, pri_res_norm, rel_error, Δtime, metric_vals, l_split; is_max_epoch=false, is_terminate_epoch=false)
    if opt.verbose > 1
        print("\n"*l_split)
        @eval(@showval("Optimizer", $method.label))
    end
    if test_model
        fvaltest = ftest(x)
        push!(fvaltests, fvaltest)
        if opt.verbose > 1
            if is_max_epoch
                max_epoch = epoch
                @show max_epoch obj fval pri_res_norm fvaltest rel_error Δtime
            elseif is_terminate_epoch
                terminate_epoch = epoch
                @show terminate_epoch obj fval pri_res_norm fvaltest rel_error Δtime
            else
                @show epoch obj fval pri_res_norm fvaltest rel_error Δtime
            end
        end
    elseif opt.verbose > 1
        if is_max_epoch
            max_epoch = epoch
            @show max_epoch obj fval pri_res_norm rel_error Δtime
        elseif is_terminate_epoch
            terminate_epoch = epoch
            @show terminate_epoch obj fval pri_res_norm rel_error Δtime
        else
            @show epoch obj fval pri_res_norm rel_error Δtime
        end
    end
    if opt.metrics !== nothing
        for name in keys(opt.metrics)
            push!(metric_vals[name], opt.metrics[name](model, x))
        end
        if opt.verbose > 1
            for name in keys(opt.metrics)
                if is_max_epoch || is_terminate_epoch
                    @eval(@showval($name, $metric_vals[$name][end]))
                else
                    @eval(@showval($name, $metric_vals[$name][$epoch+1]))
                end
            end
            if is_max_epoch || is_terminate_epoch
                print("\n"*l_split)
                if is_max_epoch
                    info_nm = opt.max_epoch > 1 ? "epochs ($(opt.max_epoch))." : "iterations ($(opt.max_iter))."
                    println("The algorithm reached its maximum number of "*info_nm)
                else
                    info_nm = opt.max_epoch > 1 ? "epoch $(opt.max_epoch)." : "iteration $(opt.max_iter)."
                    println("The algorithm terminated after a relative tolerance was reached at "*info_nm)
                end
            end
        end
    end
end

function update_stat!(objs, obj, fvals, fval, pri_res_norms, pri_res_norm, rel_errors, rel_error, f_rel_errors, f_rel_error, times, Δtime)
    push!(objs, obj)
    push!(fvals, fval)
    push!(pri_res_norms, pri_res_norm)
    push!(rel_errors, rel_error)
    push!(f_rel_errors, f_rel_error)
    push!(times, Δtime)
end