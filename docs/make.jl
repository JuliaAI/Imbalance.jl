using Documenter
using DocumenterTools: Themes
using Imbalance
using Unicode



Themes.compile(
  joinpath(@__DIR__, "src/assets/light.scss"),
  joinpath(@__DIR__, "src/assets/themes/documenter-light.css")
)



makedocs(
    sitename = "Imbalance.jl",
    authors = "Essam Wisam, mentored by Dr. Anthony Blaom",
    repo="https://github.com/JuliaAI/Imbalance.jl/",
    format = Documenter.HTML(;
    assets=[
        "assets/favicon.ico",
        asset("https://fonts.googleapis.com/css?family=Montserratwght@100;200;300;400;500;600;700;800;900|Source+Code+Pro&display=swap", class=:css)
      ]
    ),
    modules = [Imbalance],
    pages = ["Introduction" => "index.md",
              "User Guide" => "user_guide.md",
              "Algorithms" => "algorithms.md",
              "Theory" => "theory.md",
              "Examples" => "examples.md",
              "Contributing" => "contributing.md",
              "About" => "about.md"]
              
)


# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
deploydocs(repo = "github.com/JuliaAI/Imbalance.jl.git")
