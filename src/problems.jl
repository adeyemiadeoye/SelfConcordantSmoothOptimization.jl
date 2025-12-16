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
    grad_fx             # gradient of f wrt x
    hess_fx             # hessian of f wrt x
    jac_yx              # jacobian of y wrt x (used for the Prox-GGN-SCORE algorithm)
    grad_fy             # gradient of f wrt y (used for the Prox-GGN-SCORE algorithm)
    hess_fy             # hessian of f wrt y (used for the Prox-GGN-SCORE algorithm)
    name                # assigned name (optimal name for the problem)
end

mutable struct Problem <: ProxModel
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
            grad_fx::FuncOrNothing=nothing,
            hess_fx::FuncOrNothing=nothing,
            jac_yx::FuncOrNothing=nothing,
            grad_fy::FuncOrNothing=nothing,
            hess_fy::FuncOrNothing=nothing,
            name::StringOrNothing=nothing)
    return ProblemGeneric(x0, f, λ, L, sol, C_set, P, grad_fx, hess_fx, jac_yx, grad_fy, hess_fy, name)
end

function Problem(
            A::DataTupleOrArray2,
            y::VectorBitVectorOrArray2{<:Real},
            x0::Vector{Float64},
            f::Function,
            λ::IntFloatVectorOrTupleOfTwo;
            Atest::Union{DataTupleOrArray2, Nothing}=nothing,
            ytest::Union{VectorBitVectorOrArray2{<:Real}, Nothing}=nothing,
            L::Union{IntOrFloat, Nothing}=nothing,
            sol::Vector{Float64}=zero(x0),
            C_set::IntervalVectorTupleOrNothing=nothing,
            P::Union{Nothing, get_P}=nothing,
            out_fn::FuncNNOrNothing=nothing,
            grad_fx::FuncOrNothing=nothing,
            hess_fx::FuncOrNothing=nothing,
            jac_yx::FuncOrNothing=nothing,
            grad_fy::FuncOrNothing=nothing,
            hess_fy::FuncOrNothing=nothing,
            name::StringOrNothing=nothing)
    return Problem(A, y, x0, f, λ, Atest, ytest, L, sol, C_set, P, out_fn, grad_fx, hess_fx, jac_yx, grad_fy, hess_fy, name)
end

