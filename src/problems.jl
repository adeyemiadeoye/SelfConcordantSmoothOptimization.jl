using IntervalSets

export IntegerOrNothing
export IntOrFloat
export VectorOrBitVector
export ProxModel
export Problem

abstract type ProxModel end
const OperatorOrArray2 = Union{Function, AbstractArray{Float64,2}}
const StringOrNothing = Union{String, Nothing}
const FuncOrNothing = Union{Function, Nothing}
const IntegerOrNothing = Union{Integer, Nothing}
const VectorOrBitVector{T} = Union{BitVector, Vector{T}}
const IntOrFloat = Union{Int64, Float64}
const IntervalVectorTupleOrNothing = Union{ClosedInterval{T}, Interval{:closed, :closed, T}, Tuple{T, T}, Vector{T}, Nothing} where {T<:Real}

mutable struct Problem <: ProxModel
    A                   # input data
    y                   # output data
    x0                  # initial x
    f                   # smooth part of the objective function
    λ                   # penalty parameter
    L                   # gradient's Lipschitz constant or 1/α
    x                   # optimization variable (solution)
    C_set               # contains upper and lower bound of a box-constrained QP problem
    out_fn              # model output function (used for the Prox-GGN-SCORE algorithm)
    grad_fx             # gradient of f wrt x
    hess_fx             # hessian of f wrt x
    jac_yx              # jacobian of y wrt x (used for the Prox-GGN-SCORE algorithm)
    grad_fy             # gradient of f wrt y (used for the Prox-GGN-SCORE algorithm)
    hess_fy             # hessian of f wrt y (used for the Prox-GGN-SCORE algorithm)
    name                # assigned name (optimal name for the problem)
end

function Problem(
            A::OperatorOrArray2,
            y::VectorOrBitVector{<:Real},
            x0::Vector{Float64},
            f::Function,
            λ::IntOrFloat;
            Lf::Union{IntOrFloat, Nothing}=nothing,
            sol::Vector{Float64}=zero(x0),
            C_set::IntervalVectorTupleOrNothing=nothing,
            out_fn::FuncOrNothing=nothing,
            grad_fx::FuncOrNothing=nothing,
            hess_fx::FuncOrNothing=nothing,
            jac_yx::FuncOrNothing=nothing,
            grad_fy::FuncOrNothing=nothing,
            hess_fy::FuncOrNothing=nothing,
            name::StringOrNothing=nothing)
    return Problem(A, y, x0, f, λ, Lf, sol, C_set, out_fn, grad_fx, hess_fx, jac_yx, grad_fy, hess_fy, name)
end

# implement all currently supported regularization functions here
function get_reg(model::ProxModel, x, reg_name::String)
    if reg_name == "l1"
        return model.λ*sum(abs.(x))
    elseif reg_name == "l2"
        return model.λ*sum(abs2.(x))
    elseif reg_name == "indbox"
        lb, ub = minimum(model.C_set), maximum(model.C_set)
        if !(typeof(model.C_set) <: Union{ClosedInterval{<:Real}, Interval{:closed, :closed, <:Real}})
            C_set = ClosedInterval{Float64}(lb, ub)
        else
            C_set = model.C_set
        end
        return indbox_f(x, lb, ub)
        # return sum(map(x -> x ∈ C_set ? 0.0 : Inf, x))
    else
        Base.error("reg_name not valid.")
    end
end

function indbox_f(x, lb, ub)
    n = length(x)
    if any(x .< repeat([lb],n)) || any(x .> repeat([ub],n))
        return Inf
    else
        return 0.0
    end
end