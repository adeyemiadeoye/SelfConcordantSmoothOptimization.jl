import Base.show

struct Solution{A, O, R, R2, D, T, K, M}
    """
    Solution

    Stores the result of an optimization run.

    # Fields
    - `x`: Solution vector
    - `obj`: Vector of objective values per epoch/iteration
    - `fval`: Vector of function values per epoch/iteration
    - `fvaltest`: Vector of test set function values (if available)
    - `rel`: Vector of relative errors per epoch/iteration
    - `objrel`: Vector of relative objective errors per epoch/iteration
    - `metricvals`: Dictionary of custom metric values
    - `times`: Vector of elapsed times per epoch/iteration
    - `epochs`: Number of epochs/iterations performed
    - `model`: The final model or problem instance
    """
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

Base.@kwdef mutable struct Options
    metrics = nothing
    α = nothing
    batch_size = nothing
    slice_samples = false
    shuffle_batch = true
    max_epoch = 1
    max_iter = nothing
    comm_rounds = 100
    local_max_iter = nothing
    x_tol = 1e-10
    f_tol = 1e-10
    verbose = 1
end

function iter_step!(method::ProximalMethod, model::OptimModel, reg_name, hμ, As, x, x_prev, ys, Cmat, iter)
    return step!(method, model, reg_name, hμ, As, x, x_prev, ys, Cmat, iter)
end

function iterate!(method::ProximalMethod, model::SCMOModel, reg_name, hμ; metrics::Union{Dict{A,B},Nothing}=nothing, α=nothing, batch_size=nothing, slice_samples=false, shuffle_batch=true, max_epoch=1000, comm_rounds=100, local_max_iter=nothing, x_tol=1e-10, f_tol=1e-10, verbose=1) where {A,B}

    opt = Options(
            metrics=metrics,
            α=α,
            batch_size=batch_size,
            slice_samples=slice_samples,
            shuffle_batch=shuffle_batch,
            max_epoch=local_max_iter!==nothing ? 1 : max_epoch,
            comm_rounds=comm_rounds,
            local_max_iter=local_max_iter,
            x_tol=x_tol,
            f_tol=f_tol,
            verbose=verbose
        )

    solution = optim_loop!(method, model, reg_name, hμ; opt=opt)

    return solution

end

function iterate!(model::SCMOModel, reg_name, hμ; metrics::Union{Dict{A,B},Nothing}=nothing, α=nothing, batch_size=nothing, slice_samples=false, shuffle_batch=true, max_iters=1000, comm_rounds=100, local_max_iter=nothing, x_tol=1e-10, f_tol=1e-10, verbose=1) where {A,B}

    opt = Options(
            metrics=metrics,
            α=α,
            batch_size=batch_size,
            slice_samples=slice_samples,
            shuffle_batch=shuffle_batch,
            max_epoch=local_max_iter!==nothing ? 1 : max_iters,
            comm_rounds=comm_rounds,
            local_max_iter=local_max_iter,
            x_tol=x_tol,
            f_tol=f_tol,
            verbose=verbose
        )

    solution = optim_loop!(nothing, model, reg_name, hμ; opt=opt)

    return solution

end

function optim_loop!(method::ProximalMethod, model::OptimModel, reg_name, hμ; opt=Options())
    m, ny = nothing, nothing
    if typeof(model) <: ProxModel
        m = size(model.y,1)
        ny = size(model.y,2)
        if typeof(model.out_fn) <: Chain
            set_out_fn!(model)
        end
    end
    max_epoch = opt.max_epoch
    x_tol = opt.x_tol
    f_tol = opt.f_tol
    implemented_algs = []
    set_name!(method, implemented_algs)
    if opt.α !== nothing
        model.L = 1/opt.α
    end
    if method.name in implemented_algs
        if method.ss_type == 1 && model.L === nothing && opt.verbose > 0
            @info "Neither L nor α is set for the problem... Now fixing α = 0.5..."
        end
    end

    max_iter = 1
    data = [nothing]
    iend = (opt.local_max_iter !== nothing && Int(floor(opt.local_max_iter)) > 0) ? min(Int(floor(opt.local_max_iter)), max_iter) : max_iter
    if typeof(model) <: ProxModel
        max_iter = opt.batch_size !== nothing ? Int(ceil(m/opt.batch_size)) : max_iter
        iend = (opt.local_max_iter !== nothing && Int(floor(opt.local_max_iter)) > 0) ? min(Int(floor(opt.local_max_iter)), max_iter) : max_iter
        if opt.batch_size !== nothing && opt.slice_samples
            @info "Cannot use both batch_size and slice_samples=true...\nNow setting slice_samples=false..."
            opt.slice_samples = false # a.k.a prioritize mini-batching
        end
        if opt.slice_samples
            data = slice_data(model)
            opt.batch_size = 1
        elseif opt.batch_size !== nothing
            data = get_data_loader((features=model.A, targets=model.y); batchsize=opt.batch_size, shuffle=opt.shuffle_batch)
        else
            opt.batch_size = m
            data = get_data_loader((features=model.A, targets=model.y); batchsize=opt.batch_size)
        end
        data = get_loader_subset(data, 1:iend)
    end
    opt.max_iter = max_iter
    fvals = []
    fvaltests = []
    objs = []
    rel_errors = []
    f_rel_errors = []
    metric_vals = Dict()
    if opt.metrics !== nothing
        for name in keys(opt.metrics)
            metric_vals[name] = []
        end
    end
    times = []
    epochs = 0
    x_star = model.x
    if typeof(model) <: ModelGeneric
        f = (x) -> model.f(x)
        test_model = false
        ftest = (x) -> nothing
    else
        f = (x) -> model.f(model.A, model.y, x)
        test_model = all(x->x!==nothing, (model.Atest, model.ytest))
        if xor(model.Atest===nothing, model.ytest===nothing)
            @info "Both input (Atest) and target (ytest) data are required for testing the model, but only one of these has been provided.\nWill skip testing..."
        elseif test_model
            ftest = (x) -> model.f(model.Atest, model.ytest, x)
        else
            ftest = (x) -> nothing
        end
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

        if reg_name == "gl"
            Cmat = model.P.Cmat
            rel_error = mean_square_error(x_star, x)
        else
            Cmat = I
            rel_error = max(norm(x - x_star) / max.(norm(x_star),1), x_tol)
        end
        
        f_rel_error = max((norm(obj - obj_star))/norm(obj_star), f_tol)
        show_stat!(opt, model, method, x, test_model, ftest, fvaltests, epoch, obj, fval, rel_error, Δtime, metric_vals, l_split)
        update_stat!(objs, obj, fvals, fval, rel_errors, rel_error, f_rel_errors, f_rel_error, times, Δtime)

        for (i, sample) in enumerate(data)
            if typeof(model) <: ProxModel
                As, ys = sample
                (As, ys) = ny == 1 ? (Matrix(As'), vec(ys')) : (Matrix(As'), Matrix(ys'))
            else
                As, ys = nothing, nothing
            end

            if opt.verbose > 2
                if (i in [1, iend]) || (i%100==0)
                    print("\n[$i/$iend]")
                else
                    print("#")
                end
            end
            if epoch_t==max_epoch && i==iend
                Δtime = (now() - t0).value/1000
                fval = f(x)
                obj = fval + get_reg(model, x, reg_name)
                if reg_name == "gl"
                    rel_error = mean_square_error(x_star, x)
                else
                    rel_error = max(norm(x - x_star) / max.(norm(x_star),1), x_tol)
                end
                show_stat!(opt, model, method, x, test_model, ftest, fvaltests, epoch_t, obj, fval, rel_error, Δtime, metric_vals, l_split; is_max_epoch=true)
                f_rel_error = max((norm(obj - obj_star))/norm(obj_star), f_tol)
                update_stat!(objs, obj, fvals, fval, rel_errors, rel_error, f_rel_errors, f_rel_error, times, Δtime)
            end

            x_new = iter_step!(method, model, reg_name, hμ, As, x, x_prev, ys, Cmat, epoch_t)
            if norm(x_new - x) < x_tol*max.(norm(x), 1) || f_rel_error ≤ f_tol
                if epoch_t != max_epoch
                    Δtime = (now() - t0).value/1000
                    fval = f(x)
                    obj = fval + get_reg(model, x, reg_name)
                    if reg_name == "gl"
                        rel_error = mean_square_error(x_star, x)
                    else
                        rel_error = max(norm(x - x_star) / max.(norm(x_star),1), x_tol)
                    end
                    show_stat!(opt, model, method, x, test_model, ftest, fvaltests, epoch_t, obj, fval, rel_error, Δtime, metric_vals, l_split; is_terminate_epoch=true)
                    f_rel_error = max((norm(obj - obj_star))/norm(obj_star), f_tol)
                    update_stat!(objs, obj, fvals, fval, rel_errors, rel_error, f_rel_errors, f_rel_error, times, Δtime)
                end
            end
            x_prev = deepcopy(x)
            x = x_new
        end

        if norm(x - x_prev) < x_tol*max.(norm(x_prev), 1) || f_rel_error ≤ f_tol
            break
        end

        epochs += 1
        if opt.verbose > 2
            print("\n"*l_split)
        end
    end
    return Solution(x, objs, fvals, fvaltests, rel_errors, f_rel_errors, metric_vals, times, epochs, model)
end