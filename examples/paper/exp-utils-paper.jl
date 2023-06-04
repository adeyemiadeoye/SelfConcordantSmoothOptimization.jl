using SelfConcordantSmoothOptimization

using DataStructures
using StatsBase

using Plots; pyplot()
using BenchmarkProfiles

using Random
using Distributions
function rr()
    rng = MersenneTwister(1234);
    return rng
end

include("splogl1.jl")
include("spdeconv.jl")
include("benchmark-algos.jl")

markershape = [:dtriangle :rtriangle :rect :diamond :hexagon :xcross :utriangle :ltriangle :pentagon :heptagon :octagon :star4 :star6 :star7 :star8]

_plot_args = Dict{Symbol, Any}(
    :background => :white,
    :framestyle => :box,
    :grid => true,
    :gridalpha => 0.2,
    :linewidth => 2,
    :tickfontsize => 8,
    :minorgrid => true,
    :gridlinewidth => 0.7,
    :minorgridalpha => 0.06,
    :palette => :tab10,
    :background_color_legend => RGBA(1.0,1.0,1.0,0.8),
    :size => (500, 400)
)

default(; _plot_args...)

# CUSTOM MARKERS
@recipe function f(::Type{Val{:samplemarkers}}, x, y, z)
    n = length(y)
    if n > 20
        step = Int(round(n/10))
    else
        step = 1
    end
    sx, sy = x[1:step:n], y[1:step:n]
    # add an empty series with the correct type for legend markers
    @series begin
        seriestype := :path
        markershape --> :auto
        x := []
        y := []
    end
    # add a series for the line
    @series begin
        primary := false # no legend entry
        markershape := :none # ensure no markers
        seriestype := :path
        seriescolor := get(plotattributes, :seriescolor, :auto)
        x := x
        y := y
    end
    # return  a series for the sampled markers
    primary := false
    seriestype := :scatter
    markersize := 5
    
    markershape --> :auto
    x := sx
    y := sy
end

# MAIN FUNCTIONS
function solve!(model_name::String, method_name::String;
                N=2000, m=300, λ=2.0, ss_type=1, μ=2.0, max_iter=240, x_tol=1e-10, f_tol=1e-10, reg_name="l1", log_reg_problems=nothing, α=nothing, lb=-1.0, ub=1.0, verbose=0)

    #   model_name : {"sim_log", "w1a", "mushrooms", "deconv"}
    #   method_name : {"prox-ggnscore", "prox-grad", "prox-newtonscore", "prox-owlqn"}
    #   N : number of samples for synthetic data if model_name is in {"sim_log", "deconv-l1", "deconv-l2"}
    #   m : number of features for synthetic data if model_name is in {"sim_log", "deconv-l1", "deconv-l2"}
    #   λ : penalty/regularization parameter
    #   ss_type : how to select step length; 1 => use 1/Lipschitz_constant, 2=>use BB step-size, 3=>backtracking line-search
    #   μ : smoothness prameter for SCORE methods

    #   max_iter : maximum number of iterations
    #   reg_name : regularization function to use -> {"l1", "l2"} for now
    #   batch_size : batch sample size per iteration (currently only implemented for the sparse regression problems) --> nothing === "use full sample points per iteration"
    if log_reg_problems === nothing
        log_reg_problems = vcat(keys(data_dict)...,"sim_log")
    end
    if model_name in log_reg_problems # sparse logistic regression problem
        extra_metrics = false
        if reg_name == "l1"
            model = SpLogL1(model_name, N, m, λ)
            g = NormL1(λ) # g function for benchmark algos
        else
            Base.error("Please choose reg_name='l1'.")
        end
    elseif model_name == "deconv"
        extra_metrics = true
        if reg_name == "l1"
            model_name = "deconv-l1"
            g = NormL1(λ) # g function for benchmark algos
        elseif reg_name == "l2"
            model_name = "deconv-l2"
            g = NormL2(λ) # g function for benchmark algos
        else
            Base.error("Please choose reg_name in {'l1', 'l2', 'indbox'}.")
        end
        model = SpDeconv(model_name, N, λ)
    elseif model_name == "boxqp"
        extra_metrics = false
        if reg_name == "indbox"
            model = BoxQP(model_name, N, lb, ub, λ)
            g = IndBox(lb, ub) # g function for benchmark algos
        else
            Base.error("Please choose reg_name='indbox'.")
        end
    else
        Base.error("Please choose model_name in $(vcat(log_reg_problems, "deconv", "boxqp")).")
    end

    if method_name in ["panoc", "zerofpr", "f-prox-grad"]
        x0 = model.x0
        if model.L === nothing
            Base.error("Please, provide L for the becnhmark algorithms.")
        end
        L = model.L
        f = model.f
        if method_name == "panoc"
            method = BMPANOC()
        elseif method_name == "zerofpr"
            method = BMZeroFPR()
        else
            method = BMFastProxGrad()
        end
        iter_states = get_iter_states(method, x0, f, g, L)

        solution = iterate_bmark(model, iter_states, reg_name; x_tol=x_tol, f_tol=f_tol, extra_metrics=extra_metrics)
    else
        if reg_name in ["l1", "l2"]
            hμ = PHuberSmootherL1L2(μ)
        elseif reg_name == "indbox"
            hμ = ExponentialSmootherIndBox(lb, ub, μ)
        end
        if method_name == "prox-ggnscore"
            method = ProxGGNSCORE(ss_type=ss_type)
        elseif method_name == "prox-grad"
            hμ = NoSmooth(1.0)
            method = ProxGradient(ss_type=ss_type)
        elseif  method_name == "prox-newtonscore"
            method = ProxNSCORE(ss_type=ss_type)
        elseif  method_name == "prox-bfgsscore"
            method = ProxQNSCORE(ss_type=ss_type)
        elseif  method_name == "prox-owlqn"
            hμ = NoSmooth(1.0)
            method = OWLQN(ss_type=ss_type, m=20)
        else
            Base.error("Please choose method_name in {'prox-newtonscore', 'prox-ggnscore', 'panoc', 'zerofpr', 'prox-owlqn', 'prox-grad', 'f-prox-grad', 'prox-bfgsscore'}.")
        end

        solution = iterate!(method, model, reg_name, hμ; α=α, max_iter=max_iter, extra_metrics=extra_metrics, x_tol=x_tol, f_tol=f_tol, verbose=verbose)
    end

    return model, method.label, solution

end

function get_density()
    d = Dict()
    for data_name in keys(data_dict)
        N, m = data_dict[data_name][1]
        ext = data_dict[data_name][2] # file extension
        if data_name == "gisette"
            dataset_path = "examples/paper/data/"*data_name*"_train"
            A = readdlm(dataset_path*".data")
            y = vec(readdlm(dataset_path*".labels"))
        else
            dataset_path = "examples/paper/data/"*data_name*ext
            A, y = fetch_data(dataset_path, N, m)
        end
        sA = size(A)
        sA = sA[1]*sA[2]
        nz = nnz(sparse(A))
        di = nz/sA
        d[data_name] = round(di, digits=2)
    end
    return d
end

function plot_reg()
    file_dir = pwd() * "/examples/paper/figures/"
    g = x -> norm(x, 1)
    ghμ(μ) = PHuberSmootherL1L2(μ).val
    p = plot([g ghμ(0.2) ghμ(0.5) ghμ(1.0)], label=["\$g\$" "\$g □ h_{0.2}\$" "\$g □ h_{0.5}\$" "\$g □ h_{1.0}\$"], legend=:top)

    savefig(p, file_dir * "g_smooth" * ".pdf")
    return
end
    


function RUNPaperExperiments()
    problems_list = ["sim_log", "a1a", "a2a", "a3a", "a4a", "a5a", "mushrooms", "deconv"]
    log_reg_problems = problems_list[1:end-1]
    # methods_list = ["prox-newtonscore", "prox-ggnscore", "panoc", "zerofpr", "prox-owlqn", "prox-grad", "f-prox-grad", "prox-bfgsscore"]
    methods_list = ["prox-newtonscore", "prox-ggnscore", "panoc", "zerofpr", "prox-owlqn", "prox-grad", "f-prox-grad"]
    reg_list = ["l1", "l2"]

    results = []
    m = 200
    for model_name in problems_list
        for method_name in methods_list
            if model_name in log_reg_problems
                N = 2000
                reg_name = reg_list[1]
                if model_name == "sim_log"
                    λ = 2.0e-1
                    μ = 2.0
                else
                    λ = 8e-1
                    μ = 5.0
                end
                max_iter = 500
                @info "Now solving problem $model_name using $method_name..."
                model, method_label, sol = solve!(model_name, method_name; N=N, m=m, λ=λ, μ=μ, max_iter=max_iter, reg_name=reg_name, log_reg_problems=log_reg_problems)
                push!(results, (model, method_name, method_label, sol))
            else
                x_tol = 1e-6
                f_tol = 1e-6
                if method_name in ["zerofpr", "panoc", "f-prox-grad", "prox-ggnscore", "prox-grad", "prox-newtonscore"]
                    N = 512
                    λ = 2.0e-2
                    μ = 1.5
                    max_iter = 50
                    for reg_name in reg_list
                        @info "Now solving problem $model_name using $method_name, reg: $reg_name..."
                        model, method_label, sol = solve!(model_name, method_name; N=N, m=m, λ=λ, μ=μ, max_iter=max_iter, reg_name=reg_name, log_reg_problems=log_reg_problems, x_tol=x_tol, f_tol=f_tol)
                        push!(results, (model, method_name, method_label, sol))
                    end
                end
            end
        end
    end

    return results

end

function RUNPaperExperiments_α()
    file_dir = pwd() * "/examples/paper/figures/"

    problems_list = ["sim_log", "mushrooms"]
    methods_list = ["prox-newtonscore", "prox-ggnscore"]
    alpha_list = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, nothing]
    reg_name = "l1"

    N = 2000
    m = 200
    max_iter = 500
    for model_name in problems_list
        if model_name == "sim_log"
            pre_dir = "$(N)_$(m)_alpha"
            alias_name = "synthetic dataset: \$m=$N\$, \$n=$m\$"
            λ = 2.0e-1
            μ = 2.0
        else
            pre_dir = "alpha"
            alias_name = model_name * " dataset: \$m=$(data_dict[model_name][1][1])\$, \$n=$(data_dict[model_name][1][2])\$"
            λ = 8e-1
            μ = 5.0
        end

        for method_name in methods_list
            frel = []
            ts = []
            labels = []
            f_rel_ns = []
            for α in alpha_list
                @info "Now solving problem $model_name using $method_name with α = $α..."
                model, method_label, sol = solve!(model_name, method_name; N=N, m=m, λ=λ, μ=μ, max_iter=max_iter, reg_name=reg_name, log_reg_problems=problems_list, α=α)
                objrel = sol.objrel
                push!(f_rel_ns, length(objrel))
                push!(frel, objrel)
                push!(ts, sol.times)
                if α !== nothing
                    use_this_label = method_label*": \$\\alpha = $α\$"
                else
                    use_this_label = method_label*": \$\\alpha = 1/L\$"
                end
                push!(labels, use_this_label)
            end
            n = length(labels)
            labels = reshape(labels, (1,n))
            max_n = maximum(f_rel_ns)
            max_n = Int(round(max_n/min(max_n,5), digits=0))
            frelplot_alpha = plot(frel, label=labels, ylabel="\$\\|\\mathcal{L}_k-\\mathcal{L}^*\\| / \\|\\mathcal{L}^*\\|\$", xlabel="iteration number, \$k\$", yscale=:log10, title="$alias_name", titlefontsize=12, linestyle=:dashdot, ylims=(-Inf,Inf), xlims=(-Inf,Inf), xticks=0:max_n:10000)
            savefig(frelplot_alpha, file_dir * pre_dir * "_" * model_name * "_" * method_name * ".pdf")
            timesplot = plot(ts, frel, label=labels, ylabel="\$\\|\\mathcal{L}_k-\\mathcal{L}^*\\| / \\|\\mathcal{L}^*\\|\$", xlabel="time [s]", yscale=:log10, title="$alias_name", titlefontsize=12, linestyle=:dashdot, ylims=(-Inf,Inf), xlims=(-Inf,Inf))
            savefig(timesplot, file_dir * pre_dir * "_t" * "_" * model_name * "_" * method_name * ".pdf")
        end
    end

    return

end

function plot_performance_profile()
    local labels
    file_dir = pwd() * "/examples/paper/figures/"

    problems_list = keys(data_dict)
    methods_list = [Dict("prox-newtonscore"=>0.2), "prox-newtonscore", "panoc", "zerofpr", "prox-owlqn", "prox-grad", "f-prox-grad"]
    reg_name = "l1"

    n_probs = length(problems_list)

    d = get_density()
    display(d)

    T_all = Dict(name=>OrderedDict() for name in problems_list)
    It_all = Dict(name=>OrderedDict() for name in problems_list)

    T = []
    It = []

    i = 1
    max_iter = 500
    for model_name in problems_list
        labels = String[]
        if model_name == "sim_log"
            λ = 2.0e-1
            μ = 2.0
        else
            λ = 8e-1
            μ = 5.0
        end

        for method_name in methods_list
            if typeof(method_name) == Dict{String, Float64}
                α = [values(method_name)...][1]
                use_method_name = [keys(method_name)...][1]
            else
                α = nothing
                use_method_name = method_name
            end
            @info "Now solving problem $model_name using $use_method_name..."
            model, method_label, sol = solve!(model_name, use_method_name; λ=λ, μ=μ, max_iter=max_iter, reg_name=reg_name, log_reg_problems=problems_list, α=α, x_tol=1e-6, f_tol=1e-6)

            if typeof(method_name) == Dict{String, Float64}
                use_this_label = method_label*": \$\\alpha = $α\$"
            elseif method_name in ["prox-newtonscore", "prox-bfgsscore", "prox-ggnscore"]
                use_this_label = method_label*": \$\\alpha = 1/L\$"
            else
                use_this_label = method_label
            end

            push!(labels, use_this_label)

            m_time = sol.times[end]
            m_iter = float(length(sol.times))
            if i <= n_probs
                T_all[model_name][use_this_label] = [m_time]
                It_all[model_name][use_this_label] = [m_iter]
            else
                push!(T_all[model_name][use_this_label], m_time)
                push!(It_all[model_name][use_this_label], m_iter)
            end
        end
        i += 1
    end
    for label in labels
        method_times = reduce(vcat, [T_all[name][label] for name in problems_list])
        method_iters = reduce(vcat, [It_all[name][label] for name in problems_list])
        push!(T, method_times)
        push!(It, method_iters)
    end

    perfp = performance_profile(PlotsBackend(), reduce(hcat, T), labels; xlabel="\$\\tau\$", ylabel="\$\\rho(\\tau)\$", linestyle=:auto, legend=:bottomright, _plot_args...)

    savefig(perfp, file_dir * "perf_profile" * ".pdf")

    return
end

# MAIN PLOTS
function plot_allresults(results)

    local data_x, data_y, β, obj_star, x_star

    file_dir = pwd() * "/examples/paper/figures/"

    all_objp = Dict(model.name=>OrderedDict() for (model, _, _, _) in results)
    all_relp = Dict(model.name=>OrderedDict() for (model, _, _, _) in results)
    all_frelp = Dict(model.name=>OrderedDict() for (model, _, _, _) in results)
    all_timesp = Dict(model.name=>OrderedDict() for (model, _, _, _) in results)

    metrics_snrp = Dict(model.name=>OrderedDict() for (model, _, _, _) in results)
    metrics_psnrp = Dict(model.name=>OrderedDict() for (model, _, _, _) in results)
    metrics_msep = Dict(model.name=>OrderedDict() for (model, _, _, _) in results)
    metrics_rep = Dict(model.name=>OrderedDict() for (model, _, _, _) in results)
    metrics_splevelp = Dict(model.name=>OrderedDict() for (model, _, _, _) in results)
    deconv_solp = Dict(model_name=>OrderedDict() for model_name in ["deconv-l1", "deconv-l2"])

    for (model, method_name, method_label, sol) in results
        if model.name in ["deconv-l1", "deconv-l2"]
            # keep problem data
            β = model.λ
            data_x = model.x
            data_y = model.y

            metrics_snrp[model.name][method_label] = sol.metrics["snr"]
            metrics_psnrp[model.name][method_label] = sol.metrics["psnr"]
            metrics_msep[model.name][method_label] = sol.metrics["mse"]
            metrics_rep[model.name][method_label] = sol.metrics["re"]
            metrics_splevelp[model.name][method_label] = sol.metrics["sparsity_level"]
            deconv_solp[model.name][method_label] = sol.x
        end

        all_relp[model.name][method_label] = sol.rel
        all_timesp[model.name][method_label] = sol.times
        all_objp[model.name][method_label] = sol.obj
        all_frelp[model.name][method_label] = sol.objrel
    end

    for model_name in keys(all_objp)
        label = [keys(all_objp[model_name])...]
        n = length(label)
        label = reshape(label, (1,n))

        label_all = [keys(all_relp[model_name])...]
        n = length(label_all)
        label_all = reshape(label_all, (1,n))



        if model_name == "sim_log"
            alias_name = "synthetic dataset: \$m=2000\$, \$n=200\$"
        elseif model_name in ["deconv-l1", "deconv-l2"]
            alias_name = ""
        else
            alias_name = model_name * " dataset: \$m=$(data_dict[model_name][1][1])\$, \$n=$(data_dict[model_name][1][2])\$"
        end

        objs = []
        f_rel_ns = []
        for method in keys(all_objp[model_name])
            method_objs = all_objp[model_name][method]
            push!(f_rel_ns, length(method_objs))
            push!(objs, method_objs)
        end
        max_n = maximum(f_rel_ns)
        max_n = Int(round(max_n/min(max_n,5), digits=0))

        all_objplot = plot(objs, label=label, ylabel="\$\\mathcal{L}_k\$", xlabel="iteration number, \$k\$", yscale=:log10, title="$alias_name", titlefontsize=12, linestyle=:dashdot, ylims=(-Inf,Inf), xlims=(-Inf,Inf), xticks=0:max_n:10000)
        savefig(all_objplot, file_dir * "objplot" * "_" * model_name * ".pdf")

        all_frelplot = plot([all_frelp[model_name][method] for method in keys(all_frelp[model_name])], label=label, ylabel="\$\\|\\mathcal{L}_k-\\mathcal{L}^*\\| / \\|\\mathcal{L}^*\\|\$", xlabel="iteration number, \$k\$", yscale=:log10, title="$alias_name", titlefontsize=12, linestyle=:dashdot, ylims=(-Inf,Inf), xlims=(-Inf,Inf), xticks=0:max_n:10000)
        savefig(all_frelplot, file_dir * "frelplot" * "_" * model_name * ".pdf")
        
        label_t = reshape([l for l in label if l != "Prox-GGN-SCORE"], (1,n-1))
        all_timesplot = plot([all_timesp[model_name][method] for method in keys(all_timesp[model_name]) if method != "Prox-GGN-SCORE"], [all_relp[model_name][method] for method in keys(all_relp[model_name]) if method != "Prox-GGN-SCORE"], label=label_t, ylabel="\$\\|\\mathcal{L}_k-\\mathcal{L}^*\\| / \\|\\mathcal{L}^*\\|\$", xlabel="time [s]", yscale=:log10, title="$alias_name", titlefontsize=12, linestyle=:dashdot, ylims=(-Inf,Inf), xlims=(-Inf,Inf))
        savefig(all_timesplot, file_dir * "timesplot" * "_" * model_name * ".pdf")

        all_relplot = plot([all_relp[model_name][method] for method in keys(all_relp[model_name])], label=label, ylabel="\$\\frac{\\|x_k-x^*\\|}{\\max\\{\\|x^*\\|,1\\}}\$", xlabel="iteration number, \$k\$", yscale=:log10, title="$alias_name", titlefontsize=12, linestyle=:dashdot, ylims=(-Inf,Inf), xlims=(-Inf,Inf), xticks=0:max_n:10000)
        savefig(all_relplot, file_dir * "relplot" * "_" * model_name * ".pdf")


        if model_name in ["deconv-l1", "deconv-l2"]
            label = [keys(deconv_solp[model_name])...]
            n = length(label)
            label = reshape(label, (1,n))
            # snrs = [metrics_snrp[model_name][method][end] for method in keys(metrics_snrp[model_name])]
            ts = [all_timesp[model_name][method][end] for method in keys(metrics_snrp[model_name])]
            iters = [length(all_timesp[model_name][method]) for method in keys(metrics_snrp[model_name])]
            names_its_ts = [(label[i], iters[i], ts[i]) for i in 1:n]
            title = reshape(vcat(["", ""], [l*", \$\\beta = $β\$, Iter: $i, CPU time [s]: $(round(t, digits=2))" for (l,i,t) in names_its_ts]), (1, n+2))

            # plot solutions
            sol_p = [data_x, data_y]
            for method in keys(deconv_solp[model_name])
                push!(sol_p, deconv_solp[model_name][method])
            end
            # A = randn(rr(), N, m)
            deconv_solplot = plot(sol_p, layout=(Int((n+2)/2),2), label=["original" "noisy" "" "" "" "" "" ""], title=title, titlefontsize=10, size=(1000, 500), legend=:bottomright)
            savefig(deconv_solplot, file_dir * "sol" * "_" * model_name * ".pdf")

            # plot metrics
            metrics_snrplot = plot([metrics_snrp[model_name][method] for method in keys(metrics_snrp[model_name])], label=label, ylabel="SNR [dB]", xlabel="iteration number, \$k\$", size=(450, 250))
            savefig(metrics_snrplot, file_dir * "snrplot" * "_" * model_name * ".pdf")

            metrics_psnrplot = plot([metrics_psnrp[model_name][method] for method in keys(metrics_psnrp[model_name])], label=label, ylabel="PSNR [dB]", xlabel="iteration number, \$k\$", size=(450, 250))
            savefig(metrics_psnrplot, file_dir * "psnrplot" * "_" * model_name * ".pdf")

            metrics_mseplot = plot([metrics_msep[model_name][method] for method in keys(metrics_msep[model_name])], label=label, ylabel="mean squared error", xlabel="iteration number, \$k\$", size=(450, 250))
            savefig(metrics_mseplot, file_dir * "mseplot" * "_" * model_name * ".pdf")

            metrics_replot = plot([metrics_rep[model_name][method] for method in keys(metrics_rep[model_name])], label=label, ylabel="absolute error", xlabel="iteration number, \$k\$", size=(450, 250))
            savefig(metrics_replot, file_dir * "REplot" * "_" * model_name * ".pdf")

            metrics_splevelplot = plot([metrics_splevelp[model_name][method] for method in keys(metrics_splevelp[model_name])], label=label, ylabel="sparsity level", xlabel="iteration number, \$k\$", size=(450, 250))
            savefig(metrics_splevelplot, file_dir * "splevelplot" * "_" * model_name * ".pdf")
        end

    end

end


