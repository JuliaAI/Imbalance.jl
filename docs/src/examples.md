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
    <div class="grid-item">
  <a href="https://colab.research.google.com/github/JuliaAI/Imbalance.jl/blob/main/examples/effect_of_ratios.ipynb"><img id="colab" src="/examples/assets/colab.png"/></a>
  <a href="/examples/effect_of_ratios">
  <img src="/examples/assets/iris smote.jpeg" alt="Image">
  <div class="title">Effect of Ratios Hyperparameter
  <p>In this tutorial we use an SVM and SMOTE and the Iris data to study 
                      how the decision regions change with the amount of oversampling</p>
  </div>
  </a>
</div>
  <div class="grid-item">
  <a href="https://colab.research.google.com/github/JuliaAI/Imbalance.jl/blob/main/examples/effect_of_s.ipynb"><img id="colab" src="/examples/assets/colab.png"/></a>
  <a href="/examples/effect_of_s">
  <img src="/examples/assets/iris rose.jpeg" alt="Image">
  <div class="title">From Random Oversampling to ROSE
  <p>In this tutorial we study the `s` parameter in rose and the effect
                        of increasing it.</p>
  </div>
  </a>
</div>
  <div class="grid-item">
  <a href="https://colab.research.google.com/github/JuliaAI/Imbalance.jl/blob/main/examples/smote_churn_dataset.ipynb"><img id="colab" src="/examples/assets/colab.png"/></a>
  <a href="/examples/smote_churn_dataset">
  <img src="/examples/assets/churn smote.jpeg" alt="Image">
  <div class="title">SMOTE on Customer Churn Data
  <p>In this tutorial we apply SMOTE and random forest to predict customer churn based 
                        on continuous attributes.</p>
  </div>
  </a>
</div>
  <div class="grid-item">
  <a href="https://colab.research.google.com/github/JuliaAI/Imbalance.jl/blob/main/examples/smoten_mushroom.ipynb"><img id="colab" src="/examples/assets/colab.png"/></a>
  <a href="/examples/smoten_mushroom">
  <img src="/examples/assets/mushy.jpeg" alt="Image">
  <div class="title">SMOTEN on Mushroom Data
  <p>In this tutorial we use a purely categorical dataset to predict mushroom odour.</p>
  </div>
  </a>
</div>
  <div class="grid-item">
  <a href="https://colab.research.google.com/github/JuliaAI/Imbalance.jl/blob/main/examples/smotenc_churn_dataset.ipynb"><img id="colab" src="/examples/assets/colab.png"/></a>
  <a href="/examples/smotenc_churn_dataset">
  <img src="/examples/assets/churn smoten.jpeg" alt="Image">
  <div class="title">SMOTENC on Customer Churn Data
  <p>In this tutorial we extend the SMOTE tutorial to include both categorical and continuous
                        data for churn prediction</p>
  </div>
  </a>
</div>

  </div>

<script>
</body>
</html>

```
