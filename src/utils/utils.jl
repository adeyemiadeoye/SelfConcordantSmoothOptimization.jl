using StatsBase, Random

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

function update_show_stat_fed!(opt, SCSO_problem, test_model, get_metrics, metric_vals, ftest, global_x, fvaltests, round, times, Δtime; is_max_round=false)

    push!(times, Δtime)

    if test_model
        fvaltest = ftest(global_x)
        push!(fvaltests, fvaltest)
    end
    if get_metrics
        for name in keys(opt.metrics)
            push!(metric_vals[name], opt.metrics[name](SCSO_problem, global_x))
        end
    end

    if opt.verbose > 0
        l_split = "="^30*"\n"
        if is_max_round
            println(l_split*"Round $(round) completed.")
        else
            println(l_split*"Round $(round-1) completed.")
        end
        if test_model
            @show fvaltest
        end
        if get_metrics
            for name in keys(opt.metrics)
                if is_max_round
                    @eval(@showval($name, $metric_vals[$name][$round+1]))
                else
                    @eval(@showval($name, $metric_vals[$name][$round]))
                end
            end
        end
        @show Δtime
        if !(test_model || get_metrics)
            println("nothing more to show here; see docs/examples for how to define metrics to display.")
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

function set_out_fn!(model)

    x0, re_x = Flux.destructure(model.out_fn)

    function out_fn(A, x)
        flux_model = re_x(x)
        out = flux_model(A')
        return out
    end

    loss_f = deepcopy(model.f)

    loss_fn(out, y) = loss_f(out, y)
    
    function loss_fn(A,y,x)
        flux_model = re_x(x)
        out = flux_model(A')
        return loss_fn(out, y')
    end
    
    function grad_fx(A, y, x)
        flux_model = re_x(x)
        val, grads = Flux.withgradient(flux_model) do m
            loss_fn(m(A'), y')
        end
        g = Flux.destructure(grads)[1]
        return g
    end
    
    function jac_yx(A, y, out, x)
        flux_model = re_x(x)
        jac = Flux.jacobian(() -> flux_model(A'), Flux.Params(Flux.trainables(flux_model)))
        return reduce(hcat, jac)
    end
    
    function grad_fy(A, y, out)
        residual = Flux.gradient((out) -> loss_fn(out, y'), out)
        return Vector(vec(residual[1]'))
    end
    
    function hess_fy(A, y, out)
        Q = Flux.hessian((out) -> loss_fn(out, y'), out)
        return Q
    end

    if model.x0 === nothing
        model.x0 = x0
        model.sol = model.sol === nothing ? zero(x0) : model.sol
    end
    model.re = re_x
    model.out_fn = out_fn
    model.f = loss_fn
    model.grad_fx = grad_fx
    model.jac_yx = jac_yx
    model.grad_fy = grad_fy
    model.hess_fy = hess_fy

end


function set_lux_out_fn!(model)

    nnoperator = deepcopy(model.out_fn)

    ps, st = Lux.setup(Random.default_rng(model.init_seed), nnoperator)

    x0, re_x = Flux.destructure(Lux.f64(ComponentArray(ps)))

    function out_fn(A, x)
        lux_model = StatefulLuxLayer(nnoperator, re_x(x), st)
        out = lux_model(A)
        return out
    end

    loss_f = deepcopy(model.f)

    loss_fn(out, y) = loss_f(out, y)
    
    function loss_fn(A,y,x)
        lux_model = StatefulLuxLayer(nnoperator, re_x(x), st)
        out = lux_model(A)
        return loss_fn(out, y)
    end
    
    function grad_fx(A, y, x)
        g = gradient((x) -> loss_fn(A, y, x), x)
        return g
    end
    
    function jac_yx(A, y, out, x)
        jac = jacobian((x) -> out_fn(A, x), x)
        return jac
    end
    
    function grad_fy(A, y, out)
        residual = gradient((out) -> loss_fn(out, y), out)
        return Matrix(reshape(residual', (size(residual,2), size(residual,1))))
    end
    
    function hess_fy(A, y, out)
        Q = hessian((out) -> loss_fn(out, y), out)
        return Q
    end

    # if model.x0 === nothing
    #     model.x0 = x0
    #     model.x = model.x === nothing ? zero(x0) : model.x
    # end
    model.x0 = x0
    model.re = re_x
    model.out_fn = out_fn
    model.out_state = st
    model.f = loss_fn
    model.grad_fx = grad_fx
    model.jac_yx = jac_yx
    model.grad_fy = grad_fy
    model.hess_fy = hess_fy

end