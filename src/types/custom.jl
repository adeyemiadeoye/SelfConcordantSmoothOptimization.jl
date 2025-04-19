using IntervalSets
using Flux

export OptimModel
export IntegerOrNothing
export IntOrFloat
export VectorOrBitVector
export VectorBitVectorOrArray2
export VectorOrFloat
export VectorOrNothing
export is_interval_set

const OptimModel = Union{ModelGeneric, ProxModel}
const OperatorOrArray2 = Union{Function, AbstractArray{T,2}} where {T<:Real}
const VectorOrNothing = Union{Vector{T}, Nothing} where {T<:Real}
const StringOrNothing = Union{String, Nothing}
const FuncOrNothing = Union{Function, Nothing}
const FuncChainOrNothing = Union{FuncOrNothing, Chain}
const IntegerOrNothing = Union{Integer, Nothing}
const VectorOrBitVector{T} = Union{BitVector, Vector{T}}
const VectorBitVectorOrArray2{T} = Union{BitVector, Vector{T}, Matrix{T}}
const IntOrFloat = Union{Int64, Float64}
const VectorOrFloat = Union{Vector{T}, T} where {T<:Real}
const IntFloatVectorOrTupleOfTwo = Union{IntOrFloat, Tuple{T, T}, Vector{T}, Vector{Vector{T}}, Nothing} where {T<:Real}
const IntervalVectorTupleOrNothing = Union{NTuple{n, ClosedInterval{T}}, ClosedInterval{T}, Interval{:closed, :closed, T}, Tuple{T, T}, Vector{T}, Vector{Vector{T}}, Nothing} where {n, T<:Real}
const is_interval_set(x) = (typeof(x) <: Union{NTuple{n, ClosedInterval{<:Real}} where {n}, ClosedInterval{<:Real}, Interval{:closed, :closed, <:Real}})