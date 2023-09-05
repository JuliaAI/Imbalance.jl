"""
This file automatically generates the grid in examples.md from a given Julia dictionary.
"""

data = [
    Dict(
    "title" => "Effect of Ratios Hyperparameter", 
    "description" => "In this tutorial we use an SVM and SMOTE and the Iris data to study 
                      how the decision regions change with the amount of oversampling", 
    "image" => "./assets/iris smote.jpeg",
    "link" => "./effect_of_ratios",
    "colab_link" => "https://colab.research.google.com/github/JuliaAI/Imbalance.jl/blob/Doc-Examples/examples/effect_of_ratios.ipynb"
    ), 
    Dict(
      "title" => "From Random Oversampling to ROSE", 
      "description" => "In this tutorial we study the `s` parameter in rose and the effect
                        of increasing it.", 
      "image" => "./assets/iris rose.jpeg",
      "link" => "./effect_of_s",
      "colab_link" => "https://colab.research.google.com/github/JuliaAI/Imbalance.jl/blob/Doc-Examples/examples/effect_of_s.ipynb"
    ), 
    Dict(
      "title" => "SMOTE on Customer Churn Data", 
      "description" => "In this tutorial we apply SMOTE and random forest to predict customer churn based 
                        on continuous attributes.", 
      "image" => "./assets/churn smote.jpeg",
      "link" => "./smote_churn_dataset",
      "colab_link" => "https://colab.research.google.com/github/JuliaAI/Imbalance.jl/blob/Doc-Examples/examples/smote_churn_dataset.ipynb"
    ), 
    Dict(
      "title" => "SMOTEN on Mushroom Data", 
      "description" => "In this tutorial we use a purely categorical dataset to predict mushroom odour.", 
      "image" => "./assets/mushy.jpeg",
      "link" => "./smoten_mushroom",
      "colab_link" => "https://colab.research.google.com/github/JuliaAI/Imbalance.jl/blob/Doc-Examples/examples/smoten_mushroom.ipynb"
    ), 
    Dict(
      "title" => "SMOTENC on Customer Churn Data", 
      "description" => "In this tutorial we extend the SMOTE tutorial to include both categorical and continuous
                        data for churn prediction", 
      "image" => "./assets/churn smoten.jpeg",
      "link" => "./smotenc_churn_dataset",
      "colab_link" => "https://colab.research.google.com/github/JuliaAI/Imbalance.jl/blob/Doc-Examples/examples/smotenc_churn_dataset.ipynb"
    )
]


grid_items = ""
for item in data
  img_src = item["image"]
  title = item["title"]
  description = item["description"]
  link = item["link"]
  colab_link = item["colab_link"]
    grid_item = """
      <div class="grid-item">
      <a href="$colab_link"><img id="colab" src="./assets/colab.png"/></a>
      <a href="$link">
      <img src="$img_src" alt="Image">
      <div class="item-title">$title
      <p>$description</p>
      </div>
      </a>
    </div>
    """
    global grid_items *= grid_item
end

template = """
```@raw html


  <div class="grid">
  $grid_items
  </div>


```"""


output_filename = "./src/examples.md"
open(output_filename, "w") do io
    write(io, template)
end