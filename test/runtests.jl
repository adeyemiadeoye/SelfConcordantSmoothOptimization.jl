using SelfConcordantSmoothOptimization
using Test

@testset "SelfConcordantSmoothOptimization.jl" begin
    include("test_smooth.jl")
    include("test_algs.jl")
end
