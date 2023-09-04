using IntervalSets

export IntegerOrNothing
export IntOrFloat
export VectorOrBitVector
export VectorOrFloat
export is_interval_set
export bounds_sanity_check
export ProxModel
export Problem
export get_P

abstract type ProxModel end
const OperatorOrArray2 = Union{Function, AbstractArray{T,2}} where {T<:Real}
const StringOrNothing = Union{String, Nothing}
const FuncOrNothing = Union{Function, Nothing}
const IntegerOrNothing = Union{Integer, Nothing}
const VectorOrBitVector{T} = Union{BitVector, Vector{T}}
const IntOrFloat = Union{Int64, Float64}
const VectorOrFloat = Union{Vector{T}, T} where {T<:Real}
const IntFloatVectorOrTupleOfTwo = Union{IntOrFloat, Tuple{T, T}, Vector{T}, Vector{Vector{T}}, Nothing} where {T<:Real}
const IntervalVectorTupleOrNothing = Union{ClosedInterval{T}, Interval{:closed, :closed, T}, Tuple{T, T}, Vector{T}, Vector{Vector{T}}, Nothing} where {T<:Real}
const is_interval_set(x) = (typeof(x) <: Union{ClosedInterval{<:Real}, Interval{:closed, :closed, <:Real}})

mutable struct Problem <: ProxModel
    A                   # input data
    y                   # output data
    x0                  # initial x
    f                   # smooth part of the objective function
    λ                   # penalty parameter
    L                   # gradient's Lipschitz constant or 1/α
    x                   # optimization variable (solution)
    C_set               # contains upper and lower bound of a box-constrained QP problem
    Bop                 # Operator B for the fused lasso regularizer --> TODO
    P                   # Prox operator Struct for the group lasso regularizer
    out_fn              # model output function (used for the Prox-GGN-SCORE algorithm)
    grad_fx             # gradient of f wrt x
    hess_fx             # hessian of f wrt x
    jac_yx              # jacobian of y wrt x (used for the Prox-GGN-SCORE algorithm)
    grad_fy             # gradient of f wrt y (used for the Prox-GGN-SCORE algorithm)
    hess_fy             # hessian of f wrt y (used for the Prox-GGN-SCORE algorithm)
    name                # assigned name (optimal name for the problem)
end

# TODO handle this more appropriately
using SparseArrays
mutable struct get_P
    grpNUM::Int
    grpSIZES::Vector{Int}
    ntotal::Int
    ind::Matrix{Int}
    G::Vector{Int}
    matrix::SparseMatrixCSC{Int, Int}
    Cmat::Union{Matrix{Float64},UniformScaling{Bool},SparseMatrixCSC{Float64, Int64}} # A structure-aware matrix/operator
    Pi::Function
    tau::Float64
    times::Function
    trans::Function
    ProjL2::Function
    ProxL2::Function
    Lasso_fz::Function
end

function Problem(
            A::OperatorOrArray2,
            y::VectorOrBitVector{<:Real},
            x0::Vector{Float64},
            f::Function,
            λ::IntFloatVectorOrTupleOfTwo;
            Lf::Union{IntOrFloat, Nothing}=nothing,
            sol::Vector{Float64}=zero(x0),
            C_set::IntervalVectorTupleOrNothing=nothing,
            Bop::FuncOrNothing=nothing, # for fused lasso, TODO
            P::Union{Nothing, get_P}=nothing,
            out_fn::FuncOrNothing=nothing,
            grad_fx::FuncOrNothing=nothing,
            hess_fx::FuncOrNothing=nothing,
            jac_yx::FuncOrNothing=nothing,
            grad_fy::FuncOrNothing=nothing,
            hess_fy::FuncOrNothing=nothing,
            name::StringOrNothing=nothing)
    return Problem(A, y, x0, f, λ, Lf, sol, C_set, Bop, P, out_fn, grad_fx, hess_fx, jac_yx, grad_fy, hess_fy, name)
end

# implement all currently supported regularization functions here
function get_reg(model::ProxModel, x, reg_name::String)
    if reg_name == "l1" # l1 regularizer
        return model.λ*sum(abs.(x))
    elseif reg_name == "l2" # l2 regularizer
        return model.λ*sum(abs2.(x))
    elseif reg_name == "indbox" # indicator function for box constraints
        if is_interval_set(model.C_set)
            lb, ub = minimum(model.C_set), maximum(model.C_set)
        else
            lb, ub = model.C_set[1], model.C_set[2]
        end
        return indbox_f(x, lb, ub)
    elseif reg_name == "gl" # group lasso regularizer
        if length(model.λ) != 2
            Base.error("Please provide a Tuple or Vector with exactly two entries for λ, e.g. [λ1, λ2]")
        end
        P = model.P
        Px = P.matrix*(x)
        λ1, λ2 = model.λ[1], model.λ[2]
        return λ2*P.Lasso_fz(Px) + λ1*sum(abs.(x))
    else
        Base.error("reg_name not valid.")
    end
end

function indbox_f(x, lb, ub)
    if any(x .< lb) || any(x .> ub)
        return Inf
    else
        return 0.0
    end
end

function bounds_sanity_check(n, lb, ub)
    na = length(lb)
    nb = length(ub)
    if na == 1 && nb == 1
        a = repeat([lb[1]], n)
        b = repeat([ub[1]], n)
    elseif na == n && nb == n
        a = lb
        b = ub
    else
        Base.error("Lengths of the bounds do not match that of the variable.")
    end
    return a, b
end

# TODO handle this more appropriately
###################### get_P ####################
function get_P(n, G, ind)
    grpNUM = size(ind, 2)
    grpSIZES = (ind[2, :] .- ind[1, :]) .+ 1
    ntotal = sum(grpSIZES)
    Pmat = sparse(1:ntotal, G, ones(ntotal))
    
    function P_i(i)
        tmp = grpSIZES[i]
        I = 1:tmp
        J = G[ind[1, i]:ind[2, i]]
        V = ones(tmp)
        Pi = sparse(I, J, V, tmp, n)
        return Pi
    end

    Cmat = get_Cmat(ind, grpSIZES, n)
    
    P = get_P(
        grpNUM,
        grpSIZES,
        ntotal,
        ind,
        G,
        Pmat,
        Cmat,
        P_i,
        1.0,
        x -> Pmat * x,
        y -> Pmat' * y,
        (z, c1, h) -> ProjL2(z, c1, h, ind, grpNUM),
        (z, c1, h) -> ProxL2(z, c1, h, ind, grpNUM),
        z -> fz(z, ind, grpNUM)
    )
    
    return P
end

function ProjL2(x::Vector{Float64}, λ::IntOrFloat, h::Vector{Float64}, inds::Matrix{Int}, grpNUM::Int)
    m = length(x)
    Px = zeros(m)

    ind = reduce(vcat, inds)

    for j in 1:grpNUM
        βg = λ * ind[2+3*(j-1)+1]
        g_start = Int(ind[3*(j-1)+1])
        g_end = Int(ind[1+3*(j-1)+1])
        nrmval = twonorm(x ./ h, g_start, g_end)

        for k in g_start:g_end
            Px[k] = x[k] * min(βg/(h[k]*nrmval), 1)
        end
    end

    return Px
end

function ProxL2(x::Vector{Float64}, λ::IntOrFloat, h::Vector{Float64}, inds::Matrix{Int}, grpNUM::Int)
    Px = similar(x)
    ind = reduce(vcat, inds)

    for j in 1:grpNUM
        βg = λ * ind[2+3*(j-1)+1]
        g_start = Int(ind[3*(j-1)+1])
        g_end = Int(ind[1+3*(j-1)+1])

        nrmval = twonorm(x, g_start, g_end)
        for k in g_start:g_end
            Px[k] = x[k] * max(1 - βg / (h[k]*nrmval), 0)
        end
    end
    return Px
end

function fz(z::Vector{Float64}, ind::Matrix{Int}, grpNUM::Int)
    fz = 0.0
    for j in 1:grpNUM
        g_start = Int(ind[3*(j-1)+1])
        g_end = Int(ind[1+3*(j-1)+1])
        nrmval = twonorm(z, g_start, g_end)
        fz += ind[2+3*(j-1)+1]*nrmval
    end
    return fz
end

function twonorm(z::Vector{Float64}, g_start::Int, g_end::Int)
    nrm2 = 0.0
    for i in g_start:g_end
        nrm2 += z[i]^2
    end
    nrmval = sqrt(nrm2)
    return nrmval
end

function get_Cmat(ind, grpSIZES, n)
    # n: number of variables

    grpNUM = length(ind[1,:])
    g_start = ind[1,:]
    g_end = ind[2,:]
    T = zeros(Bool, grpNUM, n)
    for g = 1:grpNUM
        T[g, g_start[g]:g_end[g]] .= 1
    end
    Tw = ind[3,:]
    V, K = size(T)
    SV = sum(grpSIZES)
    J = zeros(SV)
    W = zeros(SV)
    for v = 1:V
        J[ind[1,v]:ind[2,v]] .= findall(T[v, :])
        W[ind[1,v]:ind[2,v]] .= Tw[v]
    end
    C = sparse(1:SV, J, W, SV, K)
    return C
end