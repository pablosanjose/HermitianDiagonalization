using Documenter, HermitianDiagonalization

makedocs(
    modules = [HermitianDiagonalization],
    format = :html,
    checkdocs = :exports,
    sitename = "HermitianDiagonalization.jl",
    pages = Any["index.md"]
)

deploydocs(
    repo = "github.com/pablosanjose/HermitianDiagonalization.jl.git",
)
