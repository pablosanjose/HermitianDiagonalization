struct IterativeSolvers_LOBPCG{Tv} <: AbstractEigenMethod{Tv}
    precond::Matrix{Tv}
end

IterativeSolvers_LOBPCG(h::AbstractArray{Tv}, nev = 1) where {Tv} = 
    IterativeSolvers_LOBPCG(rand(Tv, size(h,1), nev))

# defaultmethod(::Val{:IterativeSolvers}) = IterativeSolvers_LOBPCG

function (d::Diagonalizer{<:IterativeSolvers_LOBPCG, Tv})(nev::Integer, edge::SpectrumEdge = upper; 
        side = edge, precond = true, kw...) where {Tv}
    if size(d.method.precond) != (size(d.matrix, 1), nev)
        d.method = IterativeSolvers_LOBPCG(d.matrix, nev) # reset preconditioner
    end
    largest = ifelse(isfinite(d.point), side === upper , d.point > 0)
    result = IterativeSolvers.lobpcg(d.lmap, largest, d.method.precond; kw...)
    λs, ϕs = result.λ, result.X
    isfinite(d.point) && (λs .= 1 ./ λs .+ d.point)
    precond && foreach(i -> (d.method.precond[i] = ϕs[i]), eachindex(d.method.precond))
    return Eigen(real(λs), ϕs)
end