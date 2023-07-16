using Documenter
using Imbalance


# Copy the README to the home page in docs, to avoid duplication.
readme = readlines(joinpath(@__DIR__, "..", "README.md"))

open(joinpath(@__DIR__, "src/index.md"), "w") do f
    for l in readme
        println(f, l)
    end
end

makedocs(
    sitename = "Imbalance.jl",
    format = Documenter.HTML(),
    modules = [Imbalance],
    pages = ["Home" => "index.md", "API" => "api.md"],
)


# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
deploydocs(
    repo = "github.com/EssamWisam/Imbalance.jl.git"
)
