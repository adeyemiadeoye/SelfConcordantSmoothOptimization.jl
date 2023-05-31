function inv_BB_step(x, x_prev, gradx, gradx_prev)
    δ = x - x_prev
    γ = gradx - gradx_prev
    L_est = (γ ⋅ γ)/(δ' * γ) # inverse of Barzilai-Borwein (BB) step-size
    # L_est = (δ' * γ)/(δ ⋅ δ)
    return L_est
end

function linesearch(x, d, f, grad_f)
    α = 1.0
    rho = 0.5
    c = 1e-4
    while f(x + α*d) > f(x) + c*α*dot(grad_f(x), d)
        α = rho*α
    end
    return α
end

## Util functions for OWL-QN
# source: https://gist.github.com/yegortk/ce18975200e7dffd1759125972cd54f4
# (for comparison)
# projected gradient based on raw gradient, parameter values, and L1 reg. strength
function pseudo_gradient(g::Vector{Float64}, x::Vector{Float64}, λ::Float64)
    pg = zeros(size(g))
    for i in axes(g,1)
        if x[i] > 0
            pg[i] = g[i] + λ
        elseif x[i] < 0
            pg[i] = g[i] - λ
        else
            if g[i] + λ < 0
                pg[i] = g[i] + λ
            elseif g[i] - λ > 0
                pg[i] = g[i] - λ
            end
        end
    end
    return pg
end

# pi alignment operator - projection of a on orthat defined by b
function project!(a::Vector{Float64}, b::Vector{Float64})
    for i in axes(a,1)
        if sign(a[i]) != sign(b[i])
            a[i] = 0.0
        end
    end
end

# projected backtracking line search
function projected_backtracking_line_search_update(f::Function, pg::Vector{Float64}, x::Vector{Float64}, d::Vector{Float64}, λ::Float64, reg_name::String; α=1.0, β=0.5)
    gamma=1e-4

    if reg_name == "l1"
        reg_fn = x -> sum(abs.(x))
    elseif reg_name == "l2"
        reg_fn = x -> sum(abs2.(x))
    else
        Base.error("reg_name not valid.")
    end

    y = f(x) + λ*reg_fn(x)

    # choose orthant for the new point
    xi = sign.(x)
    for i in axes(xi,1)
        if xi[i] == 0
            xi[i] = sign(-pg[i])
        end
    end

    while true
        # update current point
        xt = x - α * d

        # project point onto orthant
        project!(xt, xi)

        # sufficient decrease condition
        if f(xt) + λ*reg_fn(xt) <= y + gamma * (pg ⋅ (xt - x))
            return α, xt
        end

        # update step size
        α *= β
    end
    return α, xt
end


using StatsBase
# SOME METRICS, probably not necessary (to remove)

# Signal-to-Noise Ratio (SNR)
function snr_metric(signal::Array{T,1}, noise::Array{T,1}) where T<:Real
    signal_power = sum(abs2.(signal))
    noise_power = sum(abs2.(noise))
    snr = 10 * log10(signal_power / noise_power)
    return snr
end

# Mean Squared Error (MSE)
function mean_square_error(original_signal, reconstructed_signal)
    return mean(abs2, original_signal - reconstructed_signal)
end

# Peak Signal-to-Noise Ratio (PSNR)
function psnr_metric(original_signal, reconstructed_signal; max_signal=maximum(original_signal)^2)
    mse_val = mean_square_error(original_signal, reconstructed_signal)
    return 20 * log10(max_signal) - 10 * log10(mse_val)
end

# Reconstruction Error (RE)
function recon_error(original_signal, reconstructed_signal)
    # return sum(abs2, original_signal - reconstructed_signal)
    return norm(original_signal - reconstructed_signal, 2)
end

function support_error(original_signal, reconstructed_signal; eps=1e-3)
    s = x -> abs(x)>eps ? 1 : 0
    se = s.(original_signal)
    se_hat = s.(reconstructed_signal)
    return norm(se - se_hat, 0)
end


# Sparsity Level
function sparsity_level(reconstructed_signal)
    return count(!iszero, reconstructed_signal) / length(reconstructed_signal)
end