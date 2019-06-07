struct ArnoldiMethod_IRAM{Tv} <: AbstractEigenMethod{Tv}
end

(::Type{<:ArnoldiMethod_IRAM})(h::AbstractMatrix{Tv}) where {Tv} = ArnoldiMethod_IRAM{Tv}()

# defaultmethod(::Val{:ArnoldiMethod}) = ArnoldiMethod_IRAM

function (d::Diagonalizer{<:ArnoldiMethod_IRAM,Tv})(nev::Integer; kw...) where {Tv}
    if isfinite(d.point)
        which = ArnoldiMethod.LM()
    else
        which = d.point > 0 ? ArnoldiMethod.LR() : ArnoldiMethod.SR()
    end
    decomp, _ = ArnoldiMethod.partialschur(d.lmap; nev = nev, which = which, kw...)
    λs = real.(decomp.eigenvalues)
    ϕs = decomp.Q
    isfinite(d.point) && (λs .= 1 ./ λs .+ d.point)
    return Eigen(λs, ϕs)
end