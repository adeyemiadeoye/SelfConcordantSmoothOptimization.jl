include("exp-utils-paper.jl")

@time model, _, sol = solve!("sim_log", "prox-newtonscore"; N=2000, m=200, λ=2e-1, μ=2.0, max_iter=500, reg_name="l1", ss_type=1);
# @time model, _, sol = solve!("deconv", "prox-newtonscore"; N=512, m=200, λ=2.e-2, μ=0.5, max_iter=100, reg_name="l1", ss_type=1, x_tol=1e-6, f_tol=1e-6);
# @time model, _, sol = solve!("boxqp", "prox-newtonscore"; N=60, λ=1.0, μ=1.0, max_iter=100, reg_name="indbox", ss_type=1, x_tol=1e-6, verbose=1);
# results = RUNPaperExperiments()
# plot_allresults(results)
# RUNPaperExperiments_α()

# plot_performance_profile()
# plot_reg()

;