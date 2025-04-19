import Base.show
using SelfConcordantSmoothOptimization

struct FedSolution{X, O, M, T, R}
    x::X
    fvaltest::O
    metricvals::M
    times::T
    rounds::R
end
show(io::IO, s::FedSolution) = show(io, "")

mutable struct Dataset
    features::Array{Float64, 2}
    targets::Union{Vector{Float64}, Array{Float64, 2}}
end
Base.length(d::Dataset) = size(d.features, 1)
Base.getindex(d::Dataset, i::Int) = (d.features[i, :], d.targets[i])

const DatasetOrNamedTuple = Union{Dataset, @NamedTuple{features::Matrix{Float64}, targets::Matrix{Float64}}, DataLoader}

mutable struct Client
    model::Chain
    data::DatasetOrNamedTuple
end

function optim_loop!(method::ProximalMethod, SCSO_problem_::FedModel, reg_name, hμ; opt=Options())
    SCSO_problem = deepcopy(SCSO_problem_)
    set_out_fn!(SCSO_problem)
    re_θ = SCSO_problem.re
    num_clients = length(SCSO_problem.clients_data)
    clients = [Client(deepcopy(SCSO_problem.global_model), Dataset(SCSO_problem.clients_data[i][1], SCSO_problem.clients_data[i][2])) for i in 1:num_clients]
    ftest(x) = SCSO_problem.f(SCSO_problem.Atest,  SCSO_problem.ytest, x)
    global_x = SCSO_problem.x0
    fvaltests = []
    times = []
    metric_vals = Dict()
    get_metrics = opt.metrics !== nothing
    if get_metrics
        for name in keys(opt.metrics)
            metric_vals[name] = []
        end
    end
    test_model = all(x->x!==nothing, (SCSO_problem.Atest, SCSO_problem.ytest))
    t0 = now()
    for round in 1:opt.comm_rounds
        Δtime = (now() - t0).value/1000
        # train local models
        client_models = [train_local!(method, SCSO_problem, client, reg_name, hμ, opt) for client in clients]

        update_show_stat_fed!(opt, SCSO_problem, test_model, get_metrics, metric_vals, ftest, global_x, fvaltests, round, times, Δtime)

        # federated averaging (update global model)
        global_model, global_x = fed_avg(SCSO_problem.global_model, re_θ, client_models)

        # update global model and synchronize
        SCSO_problem.x0 = global_x
        for client in clients
            client.model = global_model
        end

        if round == opt.comm_rounds
            Δtime = (now() - t0).value/1000
            update_show_stat_fed!(opt, SCSO_problem, test_model, get_metrics, metric_vals, ftest, global_x, fvaltests, round, times, Δtime; is_max_round=true)
        end
    end

    return FedSolution(global_x, fvaltests, metric_vals, times, opt.comm_rounds+1)
end

function train_local!(method::ProximalMethod, gp::FedModel, client::Client, reg_name, hμ, opt)
    train_x = client.data.features
    train_y = client.data.targets
    local_problem = Problem(train_x, train_y, gp.x0, gp.f, gp.λ; Atest=gp.Atest, ytest=gp.ytest, out_fn=client.model)

    solution = optim_loop!(method, local_problem, reg_name, hμ; opt=opt)

    return solution.x
end

function fed_avg(global_model, re, client_params)
    global_params = Flux.trainables(global_model)

    mean_cm = mean(client_params)

    for (gp, cps) in zip(global_params, Flux.trainables(re(mean_cm)))
        gp .= cps
    end

    return global_model, Flux.destructure(global_model)[1]
end