module HermitianDiagonalization

using LinearAlgebra, SparseArrays
using LinearMaps, IterativeSolvers, ArnoldiMethod, Arpack, KrylovKit, Pardiso

export diagonalizer, # reset!, 
       Direct, IRAM_Arpack, IRAM_ArnoldiMethod, IRAM_KrylovKit, LOBPCG_IterativeSolvers

ENV["OMP_NUM_THREADS"] = 4

############################################################
# LinearMap
############################################################

trivialmap(h::AbstractArray) = LinearMap(h, ishermitian = true)

function shiftinvert_linalg(h::AbstractArray{Tv}, shift) where {Tv}
    fac = lu(h - Tv(shift) * I)
    lmap = let fac = fac
        LinearMap{Tv}((x, y) -> ldiv!(x, fac, y), size(h)...,
                      ismutating = true, ishermitian = true)
    end
    return lmap, fac
end

function shiftinvert_pardiso(h::AbstractArray{Tv}, shift; verbose) where {Tv}
    ps = pardisosolver(;verbose = verbose)
    _set_matrixtype!(ps, Tv)
    hp = get_matrix(ps, h - Tv(shift) * I, :N)
    preparesolver!(ps, hp)
    lmap = let ps = ps, hp = hp
        LinearMap{Tv}((x, b) -> pardiso(ps, x, hp, b), size(h)...,
                      ismutating = true, ishermitian = true)
    end
    return lmap, ps
end

function pardisosolver(; verbose = false)
    ps = MKLPardisoSolver()
    finalizer(releasePardiso, ps)
    verbose && set_msglvl!(ps, Pardiso.MESSAGE_LEVEL_ON)
    pardisoinit(ps)
    set_iparm!(ps, 12, 2) # Pardiso expects CSR, Julia uses CSC
    return ps
end

_set_matrixtype!(ps, ::Type{<:AbstractFloat}) = set_matrixtype!(ps, Pardiso.REAL_SYM_INDEF)
_set_matrixtype!(ps, ::Type{<:Complex}) = set_matrixtype!(ps, Pardiso.COMPLEX_HERM_INDEF)

function preparesolver!(ps, hp::AbstractArray{Tv}) where {Tv}
    b = Vector{Tv}(undef, size(hp, 1))
    set_phase!(ps, Pardiso.ANALYSIS_NUM_FACT)
    pardiso(ps, hp, b)
    set_phase!(ps, Pardiso.SOLVE_ITERATIVE_REFINE)
    return ps
end

function releasePardiso(ps)
    original_phase = get_phase(ps)
    set_phase!(ps, Pardiso.RELEASE_ALL);
    pardiso(ps)
    set_phase!(ps, original_phase)
    return ps
end

############################################################
# Diagonalizers
############################################################

abstract type AbstractEigenMethod{Tv} end

mutable struct Diagonalizer{M<:AbstractEigenMethod,Tv,L<:LinearMap{Tv},C,E}
    matrix::SparseMatrixCSC{Tv, Int}
    method::M
    lmap::L
    point::Float64
    codiag::C
    engine::E   # Optional support for lmap (e.g. Pardiso solver or factorization)
end

Base.show(io::IO, d::Diagonalizer{M,Tv}) where {M,Tv} = print(io, 
"Diagonaliser{$M} for $(size(d.matrix)) Hermitian matrix around point $(d.point)")

function diagonalizer(h::AbstractArray{Tv}, ::Type{S} = defaultmethod; 
                      point = 0.0, codiag = missing, pardiso = true, 
                      verbose = false) where {Tv,S<:AbstractEigenMethod}
    ishermitian(h) || error("Matrix is non-Hermitian")
    if isfinite(point)
        lmap, engine = pardiso ? shiftinvert_pardiso(h, point; verbose = verbose) : 
                                 shiftinvert_linalg(h, point)
    else
        lmap, engine = trivialmap(h), missing
    end
    return Diagonalizer(h, S(h), lmap, point, codiag, engine)
end

# # Resets preconditioner, optionally changes method - fails to call constructor (param)
# reset!(d::Diagonalizer{M}, ::Type{M2} = M) where {M<:AbstractEigenMethod, M2<:AbstractEigenMethod} = 
#     (@show M; d.method = M2(d.lmap)) 

############################################################
# Methods
############################################################

struct Direct{Tv} <: AbstractEigenMethod{Tv}
    dense::Matrix{Tv}
end
Direct(h::AbstractArray{Tv}) where {Tv} = Direct{Tv}(Matrix(h))

struct IRAM_ArnoldiMethod{Tv} <: AbstractEigenMethod{Tv}
end
IRAM_ArnoldiMethod(h::AbstractArray{Tv}) where {Tv} = IRAM_ArnoldiMethod{Tv}()

struct IRAM_Arpack{Tv} <: AbstractEigenMethod{Tv}
    precond::Vector{Tv}
end
IRAM_Arpack(h::AbstractArray{Tv}) where {Tv} = IRAM_Arpack(rand(Tv, size(h, 2)))

struct IRAM_KrylovKit{Tv} <: AbstractEigenMethod{Tv}
    precond::Vector{Tv}
end
IRAM_KrylovKit(h::AbstractArray{Tv}) where {Tv} = IRAM_KrylovKit(rand(Tv, size(h, 2)))

struct LOBPCG_IterativeSolvers{Tv} <: AbstractEigenMethod{Tv}
    precond::Matrix{Tv}
end

LOBPCG_IterativeSolvers(h::AbstractArray{Tv}, nev = 1) where {Tv} = 
    LOBPCG_IterativeSolvers(rand(Tv, size(h,1), nev))

const defaultmethod = IRAM_KrylovKit

############################################################
# Method interfaces
############################################################

(d::Diagonalizer{<:Direct})(; kw...) = 
    eigen(Hermitian(d.method.dense); sortby = sortfunc(d.point), kw...)
function (d::Diagonalizer{<:Direct})(nev::Integer; kw...)
    e = d(; kw...)
    return Eigen(e.values[1:nev], e.vectors[:, 1:nev])
end
sortfunc(point) = isfinite(point) ? (λ -> abs(λ - point)) : (point > 0 ? reverse : identity)

function (d::Diagonalizer{<:IRAM_Arpack,Tv})(nev::Integer; kw...) where {Tv}
    if isfinite(d.point)
        which = :LM
        # sigma = real(Tv) === Tv ? d.point : d.point + 1.0im
        sigma = d.point
    elseif point > 0
        which = :LR
        sigma = nothing
    else
        which = :SR
        sigma = nothing
    end
    λs, ϕs, _ = eigs(d.matrix; nev = nev, sigma = sigma, which = which, 
                             v0 = d.method.precond, kw...)
    d.method.precond .= zero(Tv)
    foreach(ϕ -> (d.method.precond .+= ϕ), eachcol(ϕs))
    return Eigen(real(λs), ϕs)
end

function (d::Diagonalizer{<:IRAM_ArnoldiMethod,Tv})(nev::Integer; kw...) where {Tv}
    if isfinite(d.point)
        which = LM()
    else
        which = d.point > 0 ? LR() : SR()
    end
    decomp, _ = partialschur(d.lmap; nev = nev, which = which, kw...)
    λs = real.(decomp.eigenvalues)
    ϕs = decomp.Q
    isfinite(d.point) && (λs .= 1 ./ λs .+ d.point)
    return Eigen(λs, ϕs)
end

function (d::Diagonalizer{<:IRAM_KrylovKit, Tv})(nev::Integer; kw...) where {Tv}
    if isfinite(d.point)
        λs, ϕv, _ = eigsolve(x -> d.lmap * x, d.method.precond, nev; kw...)
                            #  ishermitian = true, kw...) # ishermitian fails
        λs .= 1 ./ λs .+ d.point
    else
        λs, ϕv, _ = eigsolve(d.matrix, d.method.precond, nev, d.point > 0 ? :LR : :SR, 
                             Lanczos(kw...))
    end
    d.method.precond .= zero(Tv)
    foreach(ϕ -> (d.method.precond .+= ϕ), ϕv)
    return Eigen(real(λs), hcat(ϕv))
end

function (d::Diagonalizer{<:LOBPCG_IterativeSolvers, Tv})(nev::Integer; largest = true, kw...) where {Tv}
    if size(d.method.precond) != (size(d.matrix, 1), nev)
        d.method = LOBPCG_IterativeSolvers(d.lmap, nev) # reset preconditioner
    end
    largest = ifelse(isfinite(d.point), largest, d.point > 0)
    result = lobpcg(d.lmap, I, largest, d.method.precond; kw...)
    λs, ϕs = result.λ, result.X
    isfinite(d.point) && (λs .= 1 ./ λs .+ d.point)
    foreach(i -> (d.method.precond[i] = ϕs[i]), eachindex(d.method.precond))
    return Eigen(real(λs), ϕs)
end

end # module
