module HermitianDiagonalization

using LinearAlgebra, SparseArrays
using LinearMaps, IterativeSolvers, Pardiso

ENV["OMP_NUM_THREADS"] = 4
const useMKL = Val(true)

############################################################
# Shift and invert solvers
############################################################

struct ShiftInvert{S,L}
    solver::S
    linearmap::L
end

ShiftInvert(h::AbstractArray; kw...) = ShiftInvert(h, useMKL; kw...)
function ShiftInvert(h::AbstractArray{Tv}, usepardiso::Val{true}; verbose = false) where {Tv}
    ps = MKLPardisoSolver()
    finalizer(closePardiso, ps)
    _set_matrixtype!(ps, Tv)
    verbose && set_msglvl!(ps, Pardiso.MESSAGE_LEVEL_ON)
    pardisoinit(ps)
    # set_iparm!(ps, 12, 2) # Pardiso expects CSR, Julia uses CSC
    hp = get_matrix(ps, h, :C)
    preparePardiso!(ps, hp)
    invh = LinearMap{Tv}((x, b) -> pardiso(ps, x, hp, b), size(h)...,
                      ismutating = true, ishermitian = true)
    return ShiftInvert(ps, invh)
end

function ShiftInvert(h::AbstractArray{Tv}, usepardiso::Val{false}) where {Tv}
    fac = lu(h)
    l = LinearMap{Tv}((x, y) -> ldiv!(x, fac, y), size(h)...,
                      ismutating = true, ishermitian = true)
    return ShiftInvert(fac, l)
end

_set_matrixtype!(ps, ::Type{<:AbstractFloat}) = set_matrixtype!(ps, Pardiso.REAL_SYM_INDEF)
_set_matrixtype!(ps, ::Type{<:Complex}) = set_matrixtype!(ps, Pardiso.COMPLEX_HERM_INDEF)

function preparePardiso!(ps, hp::AbstractArray{Tv}) where {Tv}
    b = Vector{Tv}(undef, size(hp, 1))
    set_phase!(ps, Pardiso.ANALYSIS_NUM_FACT)
    pardiso(ps, hp, b)
    set_phase!(ps, Pardiso.SOLVE_ITERATIVE_REFINE)
    return ps
end

closePardiso(ps) = (set_phase!(ps, Pardiso.RELEASE_ALL); pardiso(ps))

############################################################
# Shift and invert solvers
############################################################

# function diagonalize(h::AbstractArray; 
#                      check = false, nev = 6, sigma = 0.0, codiag = missing, verbose = false)
#     check && (ishermitian(h) && error("Matrix is non-Hermitian"))
#     si = linearmap(h, sigma, useMKL)
# end

# linearmap(h::AbstractArray, ::Missing, _) = LinearMap(h, ishermitian = true)
# function linearmap(h::AbstactArray, sigma::Number, ::useMKL)
#     LinearMap()
# end




end # module
