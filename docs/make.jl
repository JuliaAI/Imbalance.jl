using Documenter
using Imbalance

makedocs(
    sitename = "Imbalance",
    format = Documenter.HTML(),
    modules = [Imbalance]
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
deploydocs(
    repo = "github.com/EssamWisam/Imbalance.jl.git"
)
