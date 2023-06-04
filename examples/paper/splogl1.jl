using LinearAlgebra, SparseArrays

function rr()
    rng = MersenneTwister(1234);
    return rng
end

# SPARSE LOGISTIC REGRESSION PROBLEM

# Sparse Logistic Regression with L1 norm

function SpLogL1(data_name::String, N::IntegerOrNothing, m::IntegerOrNothing, λ::Float64)
    A, y, x0, x_star = init_Log_model(data_name; N, m) # data_name == "sim_log" or a real dataset name
    Lf = eigmax(1/N * (A'*A))
    grad_fx, hess_fx, jac_yx, grad_fy, hess_fy = get_derivative_fns_splog(A, y)
    f = x -> f_splog(A, x, y)
    return Problem(A, y, x0, f, λ; Lf=Lf, sol=x_star, out_fn=out_splog, grad_fx=grad_fx, hess_fx=hess_fx, jac_yx=jac_yx, grad_fy=grad_fy, hess_fy=hess_fy, name=data_name)
end

function f_splog(Hs::AbstractArray{S,2}, x::T, ys::B) where{S<:Real,T,B}
    n = size(ys, 1)
    return 1/n*sum(log.(1 .+ exp.(-ys .* (Hs*x))))
end

# since I am providing the derivative functions, this is not relevant
# function f_splog_y(ŷ::Array{S,1}, x::T, ys::B) where{S<:Real,T,B}
#     n = size(ys, 1)
#     return -1/n*sum(ys .* log.(ŷ) .+ (1 .- ys) .* log.(1 .- ŷ))
#     # return 1/n*sum(log.(1 .+ exp.(-ys .* ŷ)))
# end

function out_splog(A::AbstractArray{S,2}, x::T) where{S<:Real,T}
    return 1 ./ (1 .+ exp.(-A*x))
end

data_dict = Dict(
    "mushrooms" => [(8_124, 112), ".txt"],
    "a1a" => [(1_605, 123), ".txt"],
    "a2a" => [(2_265, 123), ".txt"],
    "a3a" => [(3_185, 123), ".txt"],
    "a4a" => [(4_781, 123), ".txt"],
    "a5a" => [(6_414, 123), ".txt"],
    "w1a" => [(2_477, 300), ".txt"],
    "w2a" => [(3_470, 300), ".txt"],
    "w3a" => [(4_912, 300), ".txt"],
    "w4a" => [(7_366, 300), ".txt"],
    "w5a" => [(9_888, 300), ".txt"],
    "w8a" => [(49_749, 300), ".txt"],
    "phishing" => [(11055, 68), ".txt"]
    )

function init_Log_model(data_name; N=1000, m=100)
    # N : number of samples
    # m : number of features

    if data_name == "sim_log"
        A = Matrix(sprandn(rr(), N, m, 0.75))
        true_coef = zeros(m)
        y_prob = 1 ./ (1 .+ exp.(-A * true_coef))
        y = rand.(rr(), Bernoulli.(y_prob))
        unique_y = unique(y)
        y = map(x -> x==unique_y[1] ? -1 : 1, y)
        x0 = rand(rr(), m)
        x_star = true_coef
    else
        N, m = data_dict[data_name][1]
        ext = data_dict[data_name][2] # file extension
        if ext == ".bz2"
            dataset_path = "examples/paper/data/"*data_name*"_train"
            A = readdlm(dataset_path*".data")
            y = vec(readdlm(dataset_path*".labels"))
        else
            dataset_path = "examples/paper/data/"*data_name*ext
            A, y = fetch_data(dataset_path, N, m)
        end
        unique_y = unique(y)
        y = map(x -> x==unique_y[1] ? -1 : 1, y)
        x0 = rand(rr(), m)
        x_star = zeros(m)
    end

	return A, y, x0, x_star
end

function batch_data(model::ProxModel)
    N = size(model.y, 1)
    return [(model.A[i,:],model.y[i]) for i in 1:N]
end

function sample_batch(data, m)
    # m : batch_size
    s = sample(data,m,replace=false,ordered=true)
    As = Array(hcat(map(x->x[1], s)...)')
    ys = Array(hcat(map(x->x[2], s)...)')

    return As, ys
end

# adapted from https://github.com/TheoGuyard/LIBSVMdata.jl/blob/f20a70703f16044c5c9da8046fcc0d3ca7428d00/src/LIBSVMdata.jl#L84
# the LIBSVMdata.jl package which is released under MIT license
# the main LIBSVMdata.jl package throws some HTTP error on an Ubuntu system, so I don't use it directly
function fetch_data(dataset_path, N, m; dense=true)

    # Unzip the dataset if needed
    if endswith(dataset_path, ".bz2")
        unzipped_dataset_path = string(join(split(dataset_path, ".")[begin:end-1], "."))
        display(unzipped_dataset_path)
        if !isfile(unzipped_dataset_path)
            run(`bzip2 -d -k $dataset_path`)
        end
        dataset_path = unzipped_dataset_path
    elseif endswith(dataset_path, ".tar.xz")
        unzipped_dataset_path = string(join(split(dataset_path, ".")[begin:end-2], "."))
        if !isfile(unzipped_dataset_path)
            run(`xz -d -k $dataset_path`)
        end
        dataset_path = unzipped_dataset_path
    elseif endswith(dataset_path, ".xz")
        unzipped_dataset_path = string(join(split(dataset_path, ".")[begin:end-1], "."))
        if !isfile(unzipped_dataset_path)
            run(`xz -d -k $dataset_path`)
        end
        dataset_path = unzipped_dataset_path
    end

    A = dense ? zeros(N, m) : spzeros(N, m)
    y = Vector{Float64}(undef, N)
    idx_start = 1
    open(dataset_path, "r+") do file
        lines = readlines(file)
        for (j, line) in enumerate(lines)
            elements = split(line, " ", limit=2)
            (length(elements) == 1) && push!(elements, "")
            label, features = elements
            y[j] = parse(Float64, label)
            for feature in split(features, " ")
                isempty(feature) && continue
                idx, val = split(feature, ":")
                idx = parse(Int, string(idx))
                val = parse(Float64, string(val))
                if idx == 0
                    idx_start = 0
                end
                if val != 0
                    A[j, idx-idx_start+1] = val
                end
            end
        end
    end
    return A, y
end

# get derivate functions for the for the logistic regression problem
function get_derivative_fns_splog(A::Array{Float64,2}, y::VectorOrBitVector{<:IntOrFloat})
    n = size(A, 1)
    S(x::Vector{Float64}) = exp.(-y .* (A*x))
    function grad_fx(x::Vector{Float64})
        Sx = S(x)
        return -A' * (y .* (Sx ./ (1 .+ Sx))) / n
    end
    function hess_fx(x::Vector{Float64})
        Sx = 1 ./ (1 .+ S(x))
        W = Diagonal(Sx .* (1 .- Sx))
        hess = A' * W * A
        return hess / n
    end
    # function hess_fx(x)
    #     Sx = S(x)
    #     h = 1/n * ((A .^ 2)' * (y.^2 .* (Sx ./ (1 .+ Sx))) - (A.^2)' * (y.^2 .* (Sx.^2 ./ (1 .+ Sx) .^ 2)))
    #     return Diagonal(h)
    # end
    function jac_yx(ŷ::Vector{Float64})
        return vec(ŷ .* (1.0 .- ŷ)) .* A
        # return Diagonal(ŷ .* (1.0 .- ŷ))' * A
    end
    grad_fy(y_hat::Array{Float64}) = (-y ./ y_hat .+ (1 .- y) ./ (1 .- y_hat))/n
    hess_fy(y_hat::Array{Float64}) = Diagonal((y ./ y_hat.^2 + (1 .- y) ./ (1 .- y_hat).^2)/n)
    ## or return the vector:
    # hess_fy(y_hat) = (y ./ y_hat.^2 .+ (1 .- y) ./ (1 .- y_hat).^2)./n
    
    return grad_fx, hess_fx, jac_yx, grad_fy, hess_fy
end