using Documenter
using DocumenterTools
using Imbalance
using Unicode
using Printf



include("examples.jl")

makedocs(

    sitename = "Imbalance.jl",
    authors = "Essam Wisam, mentored by Dr. Anthony Blaom",
    repo="https://github.com/JuliaAI/Imbalance.jl/",
    format = Documenter.HTML(;
    assets=[
        "assets/favicon.ico", 
        asset("https://fonts.googleapis.com/css?family=Montserratwght@100;200;300;400;500;600;700;800;900|Source+Code+Pro&display=swap", class=:css)
      ],
      repolink = "https://github.com/JuliaAI/Imbalance.jl/"
    ), 
    modules = [Imbalance], 
    pages = ["Introduction" => "index.md", 
              "Algorithms" => "algorithms.md", 
              "Walkthrough" => "walkthrough.md", 
              "Examples" => "examples.md", 
              "Contributing" => "contributing.md", 
              "About" => "about.md"],
    warnonly = true
)

#=
DocumenterTools.Themes.compile(
  joinpath(@__DIR__, "src/assets/light.scss"), 
  joinpath(@__DIR__, "src/assets/themes/documenter-light.css")
)
=#

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
deploydocs(repo = "github.com/JuliaAI/Imbalance.jl.git", devbranch = "dev")
