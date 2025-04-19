export get_reg

# all currently supported regularization functions here
function get_reg(model::OptimModel, x, reg_name::String)
    if reg_name == "l1" # l1 regularizer
        return model.λ*sum(abs.(x))
    elseif reg_name == "l2" # l2 regularizer
        return model.λ*sum(abs2.(x))
    elseif reg_name == "indbox" # indicator function for box constraints
        if is_interval_set(model.C_set)
            if isa(model.C_set, Tuple)
                lb, ub = [minimum.(model.C_set)...], [maximum.(model.C_set)...]
            else
                lb, ub = minimum(model.C_set), maximum(model.C_set)
            end
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