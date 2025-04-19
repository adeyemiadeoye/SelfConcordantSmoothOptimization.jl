μ = 1
lb = -1.0
ub = 1.0

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

@testset "Ostrovskii & Bach l1" begin
    hμ = OsBaSmootherL1L2(μ);
    @test hμ.Mh == 2*sqrt(2)
    @test hμ.ν == 3.0
end