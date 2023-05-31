
struct scaled_proximal_l1{D,T,A,S}
    model::ProxModel
    x::D
    h_scale::T
    λ::A
    α::S
end

struct scaled_proximal_l2{D,T,A,S}
    model::ProxModel
    x::D
    h_scale::T
    λ::A
    α::S
end

struct scaled_proximal_indbox{D,T,A,S}
    model::ProxModel
    x::D
    h_scale::T
    λ::A
    α::S
end

function prox_step(proxh::scaled_proximal_l1)
    (; model, x, h_scale, λ, α) = proxh
    t = α * h_scale * λ
    return sign.(x) .* max.(abs.(x) .- t, 0)
end

function prox_step(proxh::scaled_proximal_l2)
    (; model, x, h_scale, λ, α) = proxh
    t = α * h_scale * λ
    return x .* max.(1 .- t ./ abs2.(x), 0)
end

function prox_step(proxh::scaled_proximal_indbox)
    (; model, x, h_scale, λ, α) = proxh
    lb, ub = minimum(model.C_set), maximum(model.C_set)
    return min.(max.(x, lb), ub)
end

function invoke_prox(model::ProxModel, reg_name::String, x, h, λ, α)
    if reg_name == "l1"
        return scaled_proximal_l1(model, x, h, λ, α)
    elseif reg_name == "l2"
        return scaled_proximal_l2(model, x, h, λ, α)
    elseif reg_name == "indbox"
        return scaled_proximal_indbox(model, x, h, λ, α)
    else
        Base.error("reg_name not valid.")
    end
end