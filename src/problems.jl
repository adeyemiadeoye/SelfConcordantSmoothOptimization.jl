using Flux
export Problem

mutable struct ProblemLike <: ModelLike end

mutable struct ProblemGeneric <: ModelGeneric
    x0                  # initial x
    f                   # smooth part of the objective function
    λ                   # penalty parameter
    L                   # gradient's Lipschitz constant or 1/α
    x                   # optimization variable (solution)
    C_set               # contains upper and lower bound of a box-constrained QP problem
    P                   # Prox operator Struct for the group lasso regularizer
    re                  # restructure function
    grad_fx             # gradient of f wrt x
    hess_fx             # hessian of f wrt x
    jac_yx              # jacobian of y wrt x (used for the Prox-GGN-SCORE algorithm)
    grad_fy             # gradient of f wrt y (used for the Prox-GGN-SCORE algorithm)
    hess_fy             # hessian of f wrt y (used for the Prox-GGN-SCORE algorithm)
    name                # assigned name (optimal name for the problem)
end

mutable struct Problem <: ProxModel
    """
    Problem

    Represents a standard optimization problem for self-concordant smoothing algorithms.

    # Fields
    - `A`: Input data (features matrix or operator)
    - `y`: Output data (targets)
    - `x0`: Initial guess for the optimization variable
    - `f`: Smooth part of the objective function
    - `λ`: Penalty parameter(s)
    - `Atest`: Optional test input data
    - `ytest`: Optional test target data
    - `L`: Gradient's Lipschitz constant or 1/α
    - `x`: Optimization variable (solution)
    - `C_set`: Box constraint bounds
    - `P`: Prox operator struct for group lasso
    - `out_fn`: Model output function (for Prox-GGN-SCORE)
    - `re`: Restructure function (for Flux models)
    - `grad_fx`: Gradient of f with respect to x
    - `hess_fx`: Hessian of f with respect to x
    - `jac_yx`: Jacobian of y with respect to x (for Prox-GGN-SCORE)
    - `grad_fy`: Gradient of f with respect to y (for Prox-GGN-SCORE)
    - `hess_fy`: Hessian of f with respect to y (for Prox-GGN-SCORE)
    - `name`: Assigned name for the problem
    """
    A                   # input data
    y                   # output data
    x0                  # initial x
    f                   # smooth part of the objective function
    λ                   # penalty parameter
    Atest               # optional test input data
    ytest               # target for test input data
    L                   # gradient's Lipschitz constant or 1/α
    x                   # optimization variable (solution)
    C_set               # contains upper and lower bound of a box-constrained QP problem
    P                   # Prox operator Struct for the group lasso regularizer
    out_fn              # model output function (used for the Prox-GGN-SCORE algorithm)
    re                  # restructure function
    grad_fx             # gradient of f wrt x
    hess_fx             # hessian of f wrt x
    jac_yx              # jacobian of y wrt x (used for the Prox-GGN-SCORE algorithm)
    grad_fy             # gradient of f wrt y (used for the Prox-GGN-SCORE algorithm)
    hess_fy             # hessian of f wrt y (used for the Prox-GGN-SCORE algorithm)
    name                # assigned name (optimal name for the problem)
end

mutable struct FedProblem <: FedModel
    """
    FedProblem

    Represents a federated optimization problem for distributed learning.

    # Fields
    - `clients_data`: Dictionary of client datasets
    - `global_model`: Initialized global (Flux) model
    - `f`: Smooth part of the objective function
    - `λ`: Penalty parameter(s)
    - `x0`: Initial guess for the optimization variable
    - `Atest`: Optional test input data
    - `ytest`: Optional test target data
    - `L`: Gradient's Lipschitz constant or 1/α
    - `x`: Optimization variable (solution)
    - `C_set`: Box constraint bounds
    - `P`: Prox operator struct for group lasso
    - `out_fn`: Model output function (for Prox-GGN-SCORE)
    - `re`: Restructure function (for Flux models)
    - `grad_fx`: Gradient of f with respect to x
    - `hess_fx`: Hessian of f with respect to x
    - `jac_yx`: Jacobian of y with respect to x (for Prox-GGN-SCORE)
    - `grad_fy`: Gradient of f with respect to y (for Prox-GGN-SCORE)
    - `hess_fy`: Hessian of f with respect to y (for Prox-GGN-SCORE)
    - `name`: Assigned name for the problem
    """
    clients_data        # clients data
    global_model        # initialized global (Flux) model
    f                   # smooth part of the objective function
    λ                   # penalty parameter
    x0                  # initial x
    Atest               # optional test input data
    ytest               # target for test input data
    L                   # gradient's Lipschitz constant or 1/α
    x                   # optimization variable (solution)
    C_set               # contains upper and lower bound of a box-constrained QP problem
    P                   # Prox operator Struct for the group lasso regularizer
    out_fn              # model output function (used for the Prox-GGN-SCORE algorithm)
    re                  # restructure function
    grad_fx             # gradient of f wrt x
    hess_fx             # hessian of f wrt x
    jac_yx              # jacobian of y wrt x (used for the Prox-GGN-SCORE algorithm)
    grad_fy             # gradient of f wrt y (used for the Prox-GGN-SCORE algorithm)
    hess_fy             # hessian of f wrt y (used for the Prox-GGN-SCORE algorithm)
    name                # assigned name (optimal name for the problem)
end

Problem() = ProblemLike()

function Problem(
            x0::Vector{Float64},
            f::Function,
            λ::IntFloatVectorOrTupleOfTwo;
            L::Union{IntOrFloat, Nothing}=nothing,
            sol::Vector{Float64}=zero(x0),
            C_set::IntervalVectorTupleOrNothing=nothing,
            P::Union{Nothing, get_P}=nothing,
            re::Union{Nothing,Flux.Optimisers.Restructure}=nothing,
            grad_fx::FuncOrNothing=nothing,
            hess_fx::FuncOrNothing=nothing,
            jac_yx::FuncOrNothing=nothing,
            grad_fy::FuncOrNothing=nothing,
            hess_fy::FuncOrNothing=nothing,
            name::StringOrNothing=nothing)
    return ProblemGeneric(x0, f, λ, L, sol, C_set, P, re, grad_fx, hess_fx, jac_yx, grad_fy, hess_fy, name)
end

function Problem(
            A::OperatorOrArray2,
            y::VectorBitVectorOrArray2{<:Real},
            x0::Vector{Float64},
            f::Function,
            λ::IntFloatVectorOrTupleOfTwo;
            Atest::Union{OperatorOrArray2, Nothing}=nothing,
            ytest::Union{VectorBitVectorOrArray2{<:Real}, Nothing}=nothing,
            Lf::Union{IntOrFloat, Nothing}=nothing,
            sol::Vector{Float64}=zero(x0),
            C_set::IntervalVectorTupleOrNothing=nothing,
            P::Union{Nothing, get_P}=nothing,
            out_fn::FuncChainOrNothing=nothing,
            re::Union{Nothing,Flux.Optimisers.Restructure}=nothing,
            grad_fx::FuncOrNothing=nothing,
            hess_fx::FuncOrNothing=nothing,
            jac_yx::FuncOrNothing=nothing,
            grad_fy::FuncOrNothing=nothing,
            hess_fy::FuncOrNothing=nothing,
            name::StringOrNothing=nothing)
    return Problem(A, y, x0, f, λ, Atest, ytest, Lf, sol, C_set, P, out_fn, re, grad_fx, hess_fx, jac_yx, grad_fy, hess_fy, name)
end

function Problem(
    clients_data::Dict{Any, Any},
    global_model::Chain,
    f::Function,
    λ::IntFloatVectorOrTupleOfTwo;
    x0::Vector{Float64}=Flux.destructure(global_model)[1],
    Atest::Union{OperatorOrArray2, Nothing}=nothing,
    ytest::Union{VectorBitVectorOrArray2{<:Real}, Nothing}=nothing,
    Lf::Union{IntOrFloat, Nothing}=nothing,
    sol::Vector{Float64}=zero(x0),
    C_set::IntervalVectorTupleOrNothing=nothing,
    P::Union{Nothing, get_P}=nothing,
    out_fn::FuncChainOrNothing=global_model,
    re::Union{Nothing,Flux.Optimisers.Restructure}=nothing,
    grad_fx::FuncOrNothing=nothing,
    hess_fx::FuncOrNothing=nothing,
    jac_yx::FuncOrNothing=nothing,
    grad_fy::FuncOrNothing=nothing,
    hess_fy::FuncOrNothing=nothing,
    name::StringOrNothing=nothing)
    return FedProblem(clients_data, global_model, f, λ, x0, Atest, ytest, Lf, sol, C_set, P, out_fn, re, grad_fx, hess_fx, jac_yx, grad_fy, hess_fy, name)
end

