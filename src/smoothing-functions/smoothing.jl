export Smoother

abstract type Smoother end

# No smoothing
mutable struct NoSmooth <: Smoother
    μ
    val
	grad
    hess
end

NoSmooth(μ::IntOrFloat) = NoSmooth(μ, x->zero(x), x->zero(x), x->1e-6 .* one.(x))


function get_Mg(Mh::IntOrFloat, ν::IntOrFloat, μ::IntOrFloat, n::Int)
    if Mh < 0
        Base.error("Mh must be nonnegative.")
    elseif μ <= 0
        Base.error("mu must be positive.")
    end

    if 0 < ν <= 3
        return n^((3-ν)/2) * μ^(ν/2 - 2) * Mh
    elseif ν > 3
        return μ^(4 - 3*ν/2) * Mh
    else
        Base.error("ν must be positive.")
    end
end