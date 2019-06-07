module HermitianDiagonalization

using LinearAlgebra, SparseArrays, LinearMaps
using Requires

export diagonalizer, reset!, upper, lower,
       Direct, Arpack_IRAM, ArnoldiMethod_IRAM, KrylovKit_IRAM, IterativeSolvers_LOBPCG

abstract type AbstractEigenMethod{Tv} end

function __init__()
    @require Pardiso = "46dd5b70-b6fb-5a00-ae2d-e8fea33afaf2" include("pardiso.jl")
    @require KrylovKit = "0b1a1467-8014-51b9-945f-bf0ae24f4b77" include("krylovkit.jl")
    @require Arpack = "7d9fca2a-8960-54d3-9f78-7d1dccf2cb97" include("arpack.jl")
    @require ArnoldiMethod = "ec485272-7323-5ecc-a04f-4719b315124d" include("arnoldimethod.jl")
    @require IterativeSolvers = "42fd0dbc-a981-5370-80f2-aaf504508153" include("iterativesolvers.jl")
end

############################################################
# LinearMap
############################################################

struct SpectrumEdge 
    upper::Bool
end

const upper = SpectrumEdge(true)
const lower = SpectrumEdge(false)

linearmap(h::AbstractArray, ::SpectrumEdge) = LinearMap(h, ishermitian = true), missing

function linearmap(h::AbstractArray{Tv}, shift::Number) where {Tv}
    fac = lu(h - Tv(shift) * I)
    lmap = let fac = fac
        LinearMap{Tv}((x, y) -> ldiv!(x, fac, y), size(h)...,
                      ismutating = true, ishermitian = true)
    end
    return lmap, fac
end

############################################################
# Diagonalizer
############################################################

mutable struct Diagonalizer{M<:AbstractEigenMethod,Tv,A<:AbstractArray{Tv},L<:LinearMap{Tv},C,E}
    matrix::A
    method::M
    lmap::L
    point::Float64
    engine::E   # Optional support for lmap (e.g. Pardiso solver or factorization)
    codiag::C
end

Base.show(io::IO, d::Diagonalizer{M,Tv}) where {M,Tv} = print(io, 
    "Diagonaliser{$M} for $(size(d.matrix)) Hermitian matrix around point $(d.point)")

diagonalizer(h, m; kw...) = diagonalizer(h, defaultmethod(m); kw...)

function diagonalizer(h::AbstractArray{Tv}, ::Type{S} = Direct; 
                      point = 0.0, codiag = missing) where {Tv,S<:AbstractEigenMethod}
    ishermitian(h) || error("Matrix is non-Hermitian")
    lmap, engine = linearmap(h, point)
    return Diagonalizer(h, S(h), lmap, getpoint(point), engine, codiag)
end

getpoint(point::Number) = Float64(point)
getpoint(p::SpectrumEdge) = p.upper ? Inf : -Inf

reset!(d::Diagonalizer{M}) where {M} = (d.method = M(d.matrix); d)

############################################################
# Direct diagonalizer
############################################################

struct Direct{Tv} <: AbstractEigenMethod{Tv}
    dense::Matrix{Tv}
end
(::Type{<:Direct})(h::SparseMatrixCSC{Tv}) where {Tv} = Direct{Tv}(Matrix(h))

(d::Diagonalizer{<:Direct})(; kw...) = 
    eigen(Hermitian(d.method.dense); sortby = sortfunc(d.point), kw...)

function (d::Diagonalizer{<:Direct})(nev::Integer; kw...)
    e = d(; kw...)
    return Eigen(e.values[1:nev], e.vectors[:, 1:nev])
end

sortfunc(point) = isfinite(point) ? (λ -> abs(λ - point)) : (point > 0 ? reverse : identity)

reset!(d::Diagonalizer{<:Direct}) = d

############################################################
# Default defaultmethods
############################################################

# @inline defaultmethod(m::Symbol) = defaultmethod(Val(m))
# @inline defaultmethod(m::Module) = defaultmethod(Val(first(fullname(m))))

end # module
