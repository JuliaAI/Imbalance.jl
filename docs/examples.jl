"""
This file automatically generates the grid in examples.md from a given Julia dictionary.
"""

data = [
    Dict(
    "title" => "Effect of Ratios Hyperparameter", 
    "description" => "In this tutorial we use an SVM and SMOTE and the Iris data to study 
                      how the decision regions change with the amount of oversampling", 
    "image" => "/examples/assets/iris smote.jpeg",
    "link" => "/examples/effect_of_ratios",
    "colab_link" => "https://colab.research.google.com/github/JuliaAI/Imbalance.jl/blob/main/examples/effect_of_ratios.ipynb"
    ), 
    Dict(
      "title" => "From Random Oversampling to ROSE", 
      "description" => "In this tutorial we study the `s` parameter in rose and the effect
                        of increasing it.", 
      "image" => "/examples/assets/iris rose.jpeg",
      "link" => "/examples/effect_of_s",
      "colab_link" => "https://colab.research.google.com/github/JuliaAI/Imbalance.jl/blob/main/examples/effect_of_s.ipynb"
    ), 
    Dict(
      "title" => "SMOTE on Customer Churn Data", 
      "description" => "In this tutorial we apply SMOTE and random forest to predict customer churn based 
                        on continuous attributes.", 
      "image" => "/examples/assets/churn smote.jpeg",
      "link" => "/examples/smote_churn_dataset",
      "colab_link" => "https://colab.research.google.com/github/JuliaAI/Imbalance.jl/blob/main/examples/smote_churn_dataset.ipynb"
    ), 
    Dict(
      "title" => "SMOTEN on Mushroom Data", 
      "description" => "In this tutorial we use a purely categorical dataset to predict mushroom odour.", 
      "image" => "/examples/assets/mushy.jpeg",
      "link" => "/examples/smoten_mushroom",
      "colab_link" => "https://colab.research.google.com/github/JuliaAI/Imbalance.jl/blob/main/examples/smoten_mushroom.ipynb"
    ), 
    Dict(
      "title" => "SMOTENC on Customer Churn Data", 
      "description" => "In this tutorial we extend the SMOTE tutorial to include both categorical and continuous
                        data for churn prediction", 
      "image" => "/examples/assets/churn smoten.jpeg",
      "link" => "/examples/smotenc_churn_dataset",
      "colab_link" => "https://colab.research.google.com/github/JuliaAI/Imbalance.jl/blob/main/examples/smotenc_churn_dataset.ipynb"
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
      <a href="$colab_link"><img id="colab" src="/examples/assets/colab.png"/></a>
      <a href="$link">
      <img src="$img_src" alt="Image">
      <div class="title">$title
      <p>$description</p>
      </div>
      </a>
    </div>
    """
    global grid_items *= grid_item
end

template = """
```@raw html

<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<style>

  .grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
    gap: 20px;
    max-width: 1200px;
    padding: 20px;
  }

  .grid-item {
    position: relative;
    overflow: hidden;
    border-radius: 15px;
    transition: transform 0.3s ease;
    cursor: pointer;
  }

  .grid-item img {
    max-width: 100%;
    height: auto;
    display: block;
  }

  .grid-item .title {
    position: absolute;
    bottom: 0;
    left: 0;
    width: 100%;
    background-color: rgba(0, 0, 0, 0.9);
    color: #fff;
    padding: 8px 15px;
    font-size: 16px;
  }

  .title p {
    font-weight: normal !important;
    display: none;
  }



  .grid-item:hover p {
    display: block;
  }

  .grid-item:hover #colab {
    display: block;
  }

  #colab {
    border-radius: 25%;
    position: absolute;
    top: 3px;
    right: 3px;
    width: 11%;
    display: none;
  }
</style>

</head>
<body>

  <div class="grid">
  $grid_items
  </div>

<script>
</body>
</html>

```"""


output_filename = "./src/examples.md"
open(output_filename, "w") do io
    write(io, template)
end