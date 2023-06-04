μ = 0.2;
lb = -1.0;
ub = 1.0;

@testset "PHuber l1 l2" begin
    hμ = PHuberSmootherL1L2(μ);
    @test hμ.Mh == 2.0
    @test hμ.ν == 2.6
end

@testset "PHuber indbox" begin
    hμ = PHuberSmootherIndBox(lb, ub, μ);
    @test hμ.Mh == 2.0
    @test hμ.ν == 2.6
end

@testset "Exponential l1" begin
    hμ = ExponentialSmootherL1(μ);
    @test hμ.Mh == 1.0
    @test hμ.ν == 2.0
end

@testset "Exponential l2" begin
    hμ = ExponentialSmootherL2(μ);
    @test hμ.Mh == 1.0
    @test hμ.ν == 2.0
end

@testset "Exponential indbox" begin
    hμ = ExponentialSmootherIndBox(lb, ub, μ);
    @test hμ.Mh == 1.0
    @test hμ.ν == 2.0
end

@testset "Logistic l1" begin
    hμ = LogisticSmootherL1(μ);
    @test hμ.Mh == 1.0
    @test hμ.ν == 2.0
end

@testset "Burg l1" begin
    hμ = BurgSmootherL1(μ);
    @test hμ.Mh == 8.0
    @test hμ.ν == 3.0
end

@testset "Burg l2" begin
    hμ = BurgSmootherL2(μ);
    @test hμ.Mh == 8.0
    @test hμ.ν == 3.0
end

@testset "Boltzmann-Shannon l1" begin
    hμ = BoShSmootherL1(μ);
    @test hμ.Mh == 1.0
    @test hμ.ν == 4.0
end