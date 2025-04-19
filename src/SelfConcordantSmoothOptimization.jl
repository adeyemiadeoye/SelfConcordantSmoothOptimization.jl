module SelfConcordantSmoothOptimization

export ProximalMethod
export Solution
export iterate!
export NoSmooth
export PHuberSmootherL1L2, PHuberSmootherGL, PHuberSmootherIndBox
export OsBaSmootherL1L2, OsBaSmootherGL
export LogExpSmootherIndBox, LogExpSmootherIndBox2, ExponentialSmootherIndBox
export get_fed_dataset

using LinearAlgebra
using MLUtils
using ForwardDiff: gradient, hessian, jacobian
using Dates

include("types/abstract.jl")
include("types/custom.jl")
include("types/model.jl")
include("utils/utils.jl")
include("utils/prox-reg-utils.jl")
include("utils/fed-utils.jl")
include("problems.jl")
include("prox/prox-operators.jl")
include("regularizers/regularizers.jl")
include("regularizers/smoothing.jl")
include("regularizers/phuber-smooth.jl")
include("regularizers/ostrovskii-bach-smooth.jl")
include("regularizers/exponential-smooth.jl")
include("regularizers/log-exp-smooth.jl")
include("algorithms/iterate.jl")
include("algorithms/federated/fed-iterate.jl")
include("algorithms/prox-N-SCORE.jl")
include("algorithms/prox-GGN-SCORE.jl")
include("algorithms/prox-BFGS-SCORE.jl")

end