@testset "Proximal algorithms regression l1 l2" begin    
    A = [-0.560501 0.0;  0.0 1.85278; -0.0192918 -0.827763; 0.128064 0.110096; 0.0 -0.251176]
    y = [-1, -1, -1, 1, -1]
    x0 = [0.5908446386657102, 0.7667970365022592]
    λ = 1
    μ = 1
    lb = -1.0
    ub = 1.0
    f_reg(A, y, x) = 1/2*sum(log.(1 .+ exp.(-y .* (A*x))))
    f_reg(y, yhat) = -1/2*sum(y .* log.(yhat) .+ (1 .- y) .* log.(1 .- yhat))
    Mfunc(A, x) = 1 ./ (1 .+ exp.(-A*x))

    TOL = 1e-6

    @testset "Proximal Newton SCORE l1 l2" begin
        model = Problem(A, y, x0, f_reg, λ)
        sol_l1 = iterate!(ProxNSCORE(), model, "l1", PHuberSmootherL1L2(μ))
        sol_l2 = iterate!(ProxNSCORE(), model, "l2", PHuberSmootherL1L2(μ))
        @test model.x ≈ zeros(2)
        @test sol_l1.iters+1 > 1
        @test sol_l2.iters+1 > 1
        @test sol_l1.rel[end] <= TOL
        @test sol_l2.rel[end] <= TOL
        @test sol_l1.objrel[end] <= TOL
        @test sol_l2.objrel[end] <= TOL
    end

    @testset "Proximal GGN SCORE l1 l2" begin
        model = Problem(A, y, x0, f_reg, λ; out_fn=Mfunc)
        sol_l1 = iterate!(ProxGGNSCORE(), model, "l1", PHuberSmootherL1L2(μ))
        sol_l2 = iterate!(ProxGGNSCORE(), model, "l2", PHuberSmootherL1L2(μ))
        @test model.x ≈ zeros(2)
        @test sol_l1.iters+1 > 1
        @test sol_l2.iters+1 > 1
        @test sol_l1.rel[end] <= TOL
        @test sol_l2.rel[end] <= TOL
        @test sol_l1.objrel[end] <= TOL
        @test sol_l2.objrel[end] <= TOL
    end

    @testset "Proximal BFGS SCORE l1 l2" begin
        model = Problem(A, y, x0, f_reg, λ)
        sol_l1 = iterate!(ProxQNSCORE(), model, "l1", PHuberSmootherL1L2(μ))
        sol_l2 = iterate!(ProxQNSCORE(), model, "l2", PHuberSmootherL1L2(μ))
        @test model.x == zeros(2)
        @test sol_l1.iters+1 > 1
        @test sol_l2.iters+1 > 1
        @test sol_l1.rel[end] <= TOL
        @test sol_l2.rel[end] <= TOL
        @test sol_l1.objrel[end] <= TOL
        @test sol_l2.objrel[end] <= TOL
    end
end

@testset "Proximal algorithms indbox" begin    
    A =  [1.53976 0.201833 0.433995 0.156497 0.180124; 0.201833 2.37257 -0.0594941 -0.671533 0.0739676; 0.433995 -0.0594941 3.15025 0.808797 0.954656; 0.156497 -0.671533 0.808797 2.74361 0.5621; 0.180124 0.0739676 0.954656 0.5621 1.76141]
    y = [0.8673472019512456, -0.9017438158568171, -0.4944787535042339, -0.9029142938652416, 0.8644013132535154]
    x0 = [-2.07754990163271, -2.311005948690538, -0.25157276401631606, -0.8858618022602884, 1.3116613046047525]
    x_star = [-0.7139006111210786, 0.642716661564418, 0.3684773651494535, 0.5890487798472874, -0.8324174178513779]
    λ = 1.0e-4
    μ = 0.6
    lb = -1.0
    ub = 1.0
    f_qp(A, y, x) = 0.5*(x' * (A *x)) + (y'*x)

    TOL = 1e-3

    @testset "PHuber indbox" begin
        model = Problem(A, y, x0, f_qp, λ; C_set=[lb, ub], sol=x_star)
        sol_p = iterate!(ProxNSCORE(), model, "indbox", PHuberSmootherIndBox(lb, ub, μ), α=0.8)
        @test sol_p.iters+1 > 1
        @test sol_p.rel[end] <= TOL
        @test sol_p.objrel[end] <= TOL
    end

    @testset "Exp indbox" begin
        model = Problem(A, y, x0, f_qp, λ; C_set=[lb, ub], sol=x_star)
        sol_e = iterate!(ProxNSCORE(), model, "indbox", ExponentialSmootherIndBox(lb, ub, μ), α=0.8)
        @test sol_e.iters+1 > 1
        @test sol_e.rel[end] <= TOL
        @test sol_e.objrel[end] <= TOL
    end
end