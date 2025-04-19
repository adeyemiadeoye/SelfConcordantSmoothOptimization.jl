using SparseArrays

export bounds_sanity_check
export get_P

const L_INF_CACHE = -1e32
const U_INF_CACHE = 1e32

mutable struct get_P
    grpNUM::Int
    grpSIZES::Vector{Int}
    ntotal::Int
    ind::Matrix{Int}
    G::Vector{Int}
    matrix::SparseMatrixCSC{Int, Int}
    Cmat::Union{Matrix{Float64},UniformScaling{Bool},SparseMatrixCSC{Float64, Int64}} # a structure-imposing matrix/operator for group lasso
    Pi::Function
    tau::Float64
    times::Function
    trans::Function
    ProjL2::Function
    ProxL2::Function
    Lasso_fz::Function
end


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
    a[a .== -Inf] .= L_INF_CACHE
    b[b .== Inf] .= U_INF_CACHE
    return a, b
end