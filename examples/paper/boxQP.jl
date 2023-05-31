using LinearAlgebra, SparseArrays
using DSP
using IntervalSets

using Random
using Distributions
function rr()
    rng = MersenneTwister(1234);
    return rng
end


# BOX CONSTRAINED (CONVEX) QP PROBLEM
function BoxQP(model_name::String, N::Integer, λ::Float64)
    Q, q, x0, lb, ub, x_star = init_BoxQP_models(N)
    C_set = ClosedInterval{Float64}(lb, ub)
    grad_fx, hess_fx = get_derivative_fns(Q, q)
    Lf = eigmax(Q)
    f = x -> f_eval(Q, x, q)
    return Problem(Q, q, x0, f, λ; Lf=Lf, sol=x_star, C_set=C_set, out_fn=out_fn, grad_fx=grad_fx, hess_fx=hess_fx, name=model_name)
end

function f_eval(Q::AbstractArray{S,2}, x::T, ys::B) where{S,T,B}
    n = size(ys, 1)
    return 0.5*dot(x, out_fn(Q,x)) + dot(ys, x)
end

function out_fn(Q::AbstractArray{S,2}, x::T) where{S,T}
    return Q*x
end

using JuMP, Gurobi
function init_BoxQP_models(N)
    A = Matrix(sprandn(rr(), N, N, 0.60))
    q = randn(rr(), N)
    Q = 0.5*(A' * A) + 1.0e-2*I
    
    lb = 0.0
    ub = 1.0

    x0 = randn(rr(), N)

    # get an optimal solution using Gurobi global optimizer in JuMP
    model_jump = JuMP.Model(Gurobi.Optimizer)
    JuMP.set_silent(model_jump)
    @variable(model_jump, x[1:N])
    @objective(model_jump, Min, 0.5*dot(x, Q*x) + dot(q, x))
    @constraint(model_jump, lb .<= x[1:N] .<= ub)
    JuMP.optimize!(model_jump)
    x_star = JuMP.value.(x)

    return Q, q, x0, lb, ub, x_star
end

function batch_data(model::ProxModel)
    N = size(model.y, 1)
    return [(model.A[i,:],model.y[i]) for i in 1:N]
end

function sample_batch(data, m)
    # m : batch_size
    s = sample(data,m,replace=false,ordered=true)
    As = hcat(map(x->x[1], s)...)'
    ys = hcat(map(x->x[2], s)...)'

    return As, ys
end

# get derivate functions for the QP problem
function get_derivative_fns(Q::AbstractArray{Float64,2}, y::VectorOrBitVector{<:IntOrFloat})
    n = size(Q, 1)
    grad_fx(x::Vector{Float64}) = Q*x + y
    hess_fx(x::Vector{Float64}) = Q
    return grad_fx, hess_fx
end

