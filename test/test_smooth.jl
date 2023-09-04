μ = 1;
lb = -1.0;
ub = 1.0;
cd("..")
dir_ = pwd() * "/examples/paper/"
include(dir_*"spgrouplasso/spgrouplasso.jl")
A, y, x0, x_star, P = init_GroupLasso_models("sim_gl", 4, 6; grpsize=2, use_const_grpsize=true)
cd("test")
λmax = norm(A'*y,Inf)
tau = P.tau = 0.9
lambda_ = 1e-7*λmax
λ1 = tau*lambda_
λ2 = (10-tau)*lambda_
λ = [λ1,λ2]
f_reg(x) = 0.5*sum(abs2.(A*x - y))

@testset "PHuber l1 l2" begin
    hμ = PHuberSmootherL1L2(μ);
    @test hμ.Mh == 2.0
    @test hμ.ν == 2.6
end

@testset "PHuber indbox" begin
    hμ = PHuberSmootherIndBox(lb, ub, μ)
    @test hμ.Mh == 2.0
    @test hμ.ν == 2.6
end

@testset "PHuber group lasso" begin
    model = Problem(A, y, x0, f_reg, λ; P=P, sol=x_star, out_fn=(A, x)->A*x)
    hμ = PHuberSmootherGL(μ, model)
    @test hμ.Mh == 2.0
    @test hμ.ν == 2.6
end

@testset "Ostrovskii & Bach l1" begin
    hμ = OsBaSmootherL1L2(μ);
    @test hμ.Mh == 2*sqrt(2)
    @test hμ.ν == 3.0
end

@testset "Ostrovskii & Bach group lasso" begin
    model = Problem(A, y, x0, f_reg, λ; P=P, sol=x_star, out_fn=(A, x)->A*x)
    hμ = OsBaSmootherGL(μ, model)
    @test hμ.Mh == 2*sqrt(2)
    @test hμ.ν == 3.0
end