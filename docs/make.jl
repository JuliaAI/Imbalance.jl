using Documenter
using DocumenterTools
using Imbalance
using Unicode
using Printf



include("example-gen.jl")

makedocs(sitename = "Imbalance.jl",
	authors = "Essam Wisam, mentored by Dr. Anthony Blaom",
	repo = "https://github.com/JuliaAI/Imbalance.jl/",
	format = Documenter.HTML(;
		assets = [
			"assets/favicon.ico",
			asset(
				"https://fonts.googleapis.com/css?family=Montserratwght@100;200;300;400;500;600;700;800;900|Source+Code+Pro&display=swap",
				class = :css,
			),
		],
		#repolink = "https://github.com/JuliaAI/Imbalance.jl/"
	),
	modules = [Imbalance],
	pages = ["Introduction" => "index.md",
		"Algorithms" => Any[
			"Oversampling"=>"algorithms/oversampling_algorithms.md",
			"Undersampling"=>"algorithms/undersampling_algorithms.md",
      "Combination"=>"algorithms/mlj_balancing.md",
			"Extras"=>"algorithms/extra_algorithms.md",
		],
    "Tutorial" => Any[
			"Introduction"=>"examples/walkthrough.md",
			"More Examples"=>"examples.md",
		],
		"Contributing" => "contributing.md",
		"About" => "about.md"],
		warnonly=true
)



# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
deploydocs(repo = "github.com/JuliaAI/Imbalance.jl.git", devbranch = "dev")
