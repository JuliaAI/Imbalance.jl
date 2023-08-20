# Imbalance.jl

![Imbalance](assets/logos.png)
A Julia package with resampling methods to correct for class imbalance in a wide variety of classification settings.


## Motivation
Most if not all machine learning algorithms can be viewed as a form of empirical risk minimization where the object is to find the parameters $\theta$ that for some loss function $L$ minimize 

$\hat{\theta} = \arg\min_{\theta} \frac{1}{N} \sum_{i=1}^{N} L(f_{\theta}(x_i), y_i)$

where an underlying assumption is that minimizing this empirical risk corresponds to approximately minimizing the true risk which considers all examples in the populations which would imply that $f_\theta$ is approximately the true target function $f$.

In a multi-class setting, one can write

$\hat{\theta} = \arg\min_{\theta} \left( \frac{1}{N_1} \sum_{i \in \mathcal{C}_1} L(f_{\theta}(x_i), y_i) + \frac{1}{N_2} \sum_{i \in \mathcal{C}_2} L(f_{\theta}(x_i), y_i) + \ldots + \frac{1}{N_C} \sum_{i \in \mathcal{C}_C} L(f_{\theta}(x_i), y_i) \right)$

Class imbalance occurs when some classes have much fewer examples than other classes. In this case, the corresponding terms contribute minimally to the sum which makes it easier for any learning algorithm to find an approximate solution to the empirical risk that mostly only minimizes the over the significant sums. This yields a hypothesis $f_\theta$ that may be very different from the true target $f$ with respect to the minority classes which may be the most important for the application in question.

One obvious possible remedy is to weight the smaller sums so that a learning algorithm can avoid approximate solutions that exploit their insignificance. This can be shown to be equivalent (on average) to repeating examples by naive random oversampling of the observations in such classes which is offered by this package along with other more advanced oversampling methods.

To our knowledge, there are no existing maintained Julia packages that implement oversampling algorithms for multi-class classification problems or that handle both nominal and continuous features. This has served as a primary motivation for the creation of this package.

## Methods

The package so far provides five oversampling algorithms that all work in multi-class settings and with options for handling continuous and nominal features. In particular, it implements:

* Basic Random Oversampling 
* Random Sampling Examples (ROSE)
* Synthetic Minority Oversampling Technique (SMOTE)
* SMOTE-Nominal (SMOTE-N)
* SMOTE-Nominal Categorical (SMOTE-NC)

## Features
- Provides some of the most sought oversampling algorithms in machine learning and is still under development
- Supports multi-class classification and both nominal and continuous features
- Generic by supporting table input/output formats as well as matrices
- Integrates seamlessly with other JuliaAI or JuliaML packages such as `MLJ` and `TableTransforms` by implementing their respective interfaces
- Supports tables regardless to whether `y` is a separate column or one of the columns
- Supports automatic encoding and decoding of nominal features

## Installation
```julia
import Pkg;
Pkg.add("Imbalance")
```

## Quick Start
We will illustrate this with `SMOTE` but all other methods follow the same pattern.

### Standard API
```julia
using Imbalance

# Set dataset properties then generate imbalanced data
probs = [0.5, 0.2, 0.3]                  # probability of each class      
num_rows, num_cont_feats = 100, 5
X, y = generate_imbalanced_data(num_rows, num_cont_feats; probs, rng=42)      

# Apply SMOTE to oversample the classes
Xover, yover = smote(X, y; k=5, ratios=Dict(0=>1.0, 1=> 0.9, 2=>0.8), rng=42)

```
### MLJ Interface
All methods support the `MLJ` interface over tables where instead of directly calling the method, one instantiates a model for the method while passing any necessary keyword parameters, wraps the model in a machine and then calls transform on the machine and data.
```julia
using MLJ

# Load the model
SMOTE = @load SMOTE pkg=Imbalance

# Create an instance of the model 
oversampler = SMOTE(k=5, ratios=Dict(0=>1.0, 1=> 0.9, 2=>0.8), rng=42)

# Wrap it in a machine
mach = machine(oversampler)

# Provide the data to transform 
Xover, yover = transform(mach, X, y)
```
All methods are considered static transforms and hence, no `fit` is required. 

### Table Transforms Interface
This interface operates on single tables. It thereby assumes that `y` is one of the columns involved. Thus, it follows a similar pattern to the `MLJ` interface except that the index of `y` is a required argument while instantiating the model and the data to be transformed is only one table `Xy`.
```julia
using Imbalance
using TableTransforms

# Generate imbalanced data
num_rows = 200
num_features = 5
y_ind = 3
Xy, _ = generate_imbalanced_data(num_rows, num_features; 
                                 probs=[0.5, 0.2, 0.3], insert_y=y_ind, rng=42)

# Initiate SMOTE model
oversampler = SMOTE_t(y_ind; k=5, ratios=Dict(0=>1.0, 1=> 0.9, 2=>0.8), rng=42)
Xyover = Xy |> oversampler                                
Xyover, cache = TableTransforms.apply(oversampler, Xy)    # equivalently
```
The `reapply(oversampler, Xy, cache)` method from `TableTransforms` simply falls back to `apply(oversample, Xy)` and the `revert(oversampler, Xy, cache)` reverts the transform by removing the oversampled observations from the table.
