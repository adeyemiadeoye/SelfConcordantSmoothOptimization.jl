include("exp-utils-paper.jl");
using NPZ

# Run the paper experiments and generate plots
# uncomment the desired line to run


##### RUN INDIVIDUAL EXAMPLE #####
# adjust the parameters according to paper

@time model, _, sol = solve!("sim_log", "prox-ggnscore"; m=100, n=1000, λ=1, μ=1, max_iter=2000, reg_name="l1", ss_type=1, verbose=2, α=1);

# @time model, _, sol = solve!("deconv", "prox-grad"; m=1024, n=200, λ=1e-3, μ=5e-2, max_iter=5000, reg_name="l1", ss_type=1, x_tol=1e-6, f_tol=1e-6, verbose=2, α=1);

# @time model, _, sol = solve!("boxqp", "prox-newtonscore"; m=200, λ=1.0e-4, μ=0.6, reg_name="indbox", max_iter=10000, x_tol=1e-10, f_tol=1e-10, α=0.8, ss_type=1, lb=-1.0, ub=1.0, verbose=0);

# @time model, _, sol = solve!("fusedlasso", "prox-ggnscore"; m=100, n=1000, λ=(1e-7, 1e-3), μ=5.0, max_iter=20000, reg_name="fl", ss_type=1, verbose=2, x_tol=1e-8);

# m, n = 500, 2000
# problem = "sim_gl"
# method = "prox-ggnscore"
# if method in ["prox-ggnscore", "prox-newtonscore", "prox-bfgsscore"]
#     α = 1
#     λ = 1e-8
# else
#     # let other algorithms use L for α
#     # better results with λ = 1e-7
#     α = nothing
#     λ = 1e-7
# end
# @time model, _, sol = solve!(problem, method; m=m, n=n, λ=λ, μ=1.6, max_iter=100000, reg_name="gl", grpsize=100, use_const_grpsize=true, ss_type=1, verbose=2, x_tol=1e-9, α=α);



##### RUN ALL THE EXAMPLES AND PLOT #####
# uncomment to run

# results = RUNPaperExperiments();

# plot_allresults(results);

# RUNPaperExperiments_α(4000, 100);

# result, x = RUNPaperExperiments_SGL(1000, 10000);

# plot_performance_profile(n_runs=20);

# plot_reg(ex="1"); # ex="1" or "2" to generate Example 1 and Example 2 figures in Section 2

;