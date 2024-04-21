using StatsBase
function mean_square_error(original_signal, reconstructed_signal)
    return mean(abs2, original_signal - reconstructed_signal)
end

macro showval(name, expression)
    quote
        value = $expression
        println($name, ":\t", value)
    end
end

function slice_data(model::ProxModel)
    return zip(eachslice(model.A; dims=1), eachrow(model.y))
end