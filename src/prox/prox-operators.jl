struct scaled_proximal_l1{D,T,A,S}
    model::SCMOModel
    x::D
    h_scale::T
    λ::A
    α::S
end
function prox_step(proxh::scaled_proximal_l1)
    (; model, x, h_scale, λ, α) = proxh
    t = α * λ ./ h_scale
    return sign.(x) .* max.(abs.(x) .- t, 0)
end

struct scaled_proximal_l2{D,T,A,S}
    model::SCMOModel
    x::D
    h_scale::T
    λ::A
    α::S
end
function prox_step(proxh::scaled_proximal_l2)
    (; model, x, h_scale, λ, α) = proxh
    t = α * λ ./ h_scale
    return x .* max.(1 .- t ./ abs2.(x), 0)
end

struct scaled_proximal_indbox{D,T,A,S}
    model::SCMOModel
    x::D
    h_scale::T
    λ::A
    α::S
end
function prox_step(proxh::scaled_proximal_indbox)
    (; model, x, h_scale, λ, α) = proxh
    if is_interval_set(model.C_set)
        if isa(model.C_set, Tuple)
            lb, ub = [minimum.(model.C_set)...], [maximum.(model.C_set)...]
        else
            lb, ub = minimum(model.C_set), maximum(model.C_set)
        end
    else
        lb, ub = model.C_set[1], model.C_set[2]
    end
    return min.(max.(x, lb), ub)
end

struct scaled_proximal_grouplasso{D,T,A,S}
    model::SCMOModel
    x::D
    h_scale::T
    λ::A
    α::S
end
function prox_step(proxh::scaled_proximal_grouplasso)
    (; model, x, h_scale, λ, α) = proxh
    # get the true value of λ from the model
    P = model.P
    λ1, λ2 = model.λ[1], model.λ[2]
    t = λ1 ./ h_scale

    utmp = sign.(x) .* max.(abs.(x) .- t, 0) # ProxL1
    u = P.ProxL2(utmp, α*λ2, h_scale) # ProxL2

    return u
end

function invoke_prox(model::SCMOModel, reg_name::String, x, h, λ, α)
    if reg_name == "l1"
        return scaled_proximal_l1(model, x, h, λ, α)
    elseif reg_name == "l2"
        return scaled_proximal_l2(model, x, h, λ, α)
    elseif reg_name == "indbox"
        return scaled_proximal_indbox(model, x, h, λ, α)
    elseif reg_name == "gl"
        return scaled_proximal_grouplasso(model, x, h, λ, α)
    else
        Base.error("reg_name not valid.")
    end
end