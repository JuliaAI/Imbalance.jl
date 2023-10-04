# Imbalance.jl
![Imbalance](https://i.imgur.com/CP97JdN.png)


A Julia package with resampling methods to correct for class imbalance in a wide variety of classification settings.


## Installation
```julia
import Pkg;
Pkg.add("Imbalance")
```

## Implemented Methods

The package implements the following resampling algorithms

- Random Oversampling
- Random Walk Oversampling (RWO)
- Random Oversampling Examples (ROSE)
- Synthetic Minority Oversampling Technique (SMOTE)
- Borderline SMOTE1
- SMOTE-Nominal (SMOTE-N)
- SMOTE-Nominal Categorical (SMOTE-NC)
- Random Undersampling
- Cluster Undersampling
- EditedNearestNeighbors Undersampling
- Tomek Links Undersampling
- Balanced Bagging Classifier (@MLJBalancing.jl)

Interested in contributing with more? Check [this](https://juliaai.github.io/Imbalance.jl/dev/contributing/).

## Quick Start

We will illustrate using the package to oversample with`SMOTE`; however, all other implemented oversampling methods follow the same pattern.

### Standard API
All methods by default support a pure functional interface.
```julia
using Imbalance

# Set dataset properties then generate imbalanced data
class_probs = [0.5, 0.2, 0.3]                  # probability of each class      
num_rows, num_continuous_feats = 100, 5
X, y = generate_imbalanced_data(num_rows, num_continuous_feats; class_probs, rng=42)      

# Apply SMOTE to oversample the classes
Xover, yover = smote(X, y; k=5, ratios=Dict(0=>1.0, 1=> 0.9, 2=>0.8), rng=42)

```

### MLJ Interface
All methods support the [`MLJ` interface](https://alan-turing-institute.github.io/MLJ.jl/dev/) where instead of directly calling the method, one instantiates a model for the method while optionally passing the keyword parameters found in the functional interface then wraps the model in a `machine` and follows by calling `transform` on the machine and data.
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
All implemented oversampling methods are considered static transforms and hence, no `fit` is required. 

#### Pipelining Models
If `MLJBalancing` is also used, an arbitrary number of resampling methods from `Imbalance.jl` can be wrapped with a classification model from `MLJ` to function as a unified model where resampling automatically takes place on given data before training the model.

```julia
using MLJBalancing

# grab one more resampler and a classifier
LogisticClassifier = @load LogisticClassifier pkg=MLJLinearModels verbosity=0
TomekUndersampler = @load TomekUndersampler pkg=Imbalance verbosity=0

undersampler = TomekUndersampler(min_ratios=0.5, rng=42)
logistic_model = LogisticClassifier()

# wrap the oversampler, undersample and classification model together
balanced_model = BalancedModel(model=logistic_model, 
                               balancer1=oversampler, balancer2=undersampler)

# behaves like a single model
mach = machine(balanced_model, X, y);
fit!(mach, verbosity=0)
predict(mach, X)
```

### Table Transforms Interface
The [`TableTransforms` interface](https://juliaml.github.io/TableTransforms.jl/stable/transforms/) operates on single tables; it assumes that `y` is one of the columns of the given table. Thus, it follows a similar pattern to the `MLJ` interface except that the index of `y` is a required argument while instantiating the model and the data to be transformed via `apply` is only one table `Xy`.
```julia
using Imbalance
using Imbalance.TableTransforms
using TableTransforms

# Generate imbalanced data
num_rows = 200
num_features = 5
y_ind = 3
Xy, _ = generate_imbalanced_data(num_rows, num_features; 
                                 class_probs=[0.5, 0.2, 0.3], insert_y=y_ind, rng=42)

# Initiate SMOTE model
oversampler = SMOTE(y_ind; k=5, ratios=Dict(0=>1.0, 1=> 0.9, 2=>0.8), rng=42)
Xyover = Xy |> oversampler       # can chain with other table transforms                  
# equivalently if TableTransforms is used
Xyover, cache = TableTransforms.apply(oversampler, Xy)    # equivalently
```
The `reapply(oversampler, Xy, cache)` method from `TableTransforms` simply falls back to `apply(oversample, Xy)` and the `revert(oversampler, Xy, cache)` reverts the transform by removing the oversampled observations from the table.


## Features
- Supports multi-class variants of the algorithms and both nominal and continuous features
- Provides `MLJ` and `TableTransforms` interfaces aside from the default pure functional interface
- Generic by supporting table input/output formats as well as matrices
- Supports tables regardless to whether the target is a separate column or one of the columns
- Supports automatic encoding and decoding of nominal features


## Rationale
Most if not all machine learning algorithms can be viewed as a form of empirical risk minimization where the object is to find the parameters $\theta$ that for some loss function $L$ minimize 

$$\hat{\theta} = \arg\min_{\theta} \frac{1}{N} \sum_{i=1}^{N} L(f_{\theta}(x_i), y_i)$$

The underlying assumption is that minimizing this empirical risk corresponds to approximately minimizing the true risk which considers all examples in the populations which would imply that $f_\theta$ is approximately the true target function $f$ that we seek to model.

In a multi-class setting with $K$ classes, one can write

$$\hat{\theta} = \arg\min_{\theta} \left( \frac{1}{N_1} \sum_{i \in C_1} L(f_{\theta}(x_i), y_i) + \frac{1}{N_2} \sum_{i \in C_2} L(f_{\theta}(x_i), y_i) + \ldots + \frac{1}{N_K} \sum_{i \in C_K} L(f_{\theta}(x_i), y_i) \right)$$

Class imbalance occurs when some classes have much fewer examples than other classes. In this case, the terms corresponding to smaller classes contribute minimally to the sum which makes it possible for any learning algorithm to find an approximate solution to minimizing the empirical risk that mostly only minimizes the over the significant sums. This yields a hypothesis $f_\theta$ that may be very different from the true target $f$ with respect to the minority classes which may be the most important for the application in question.

One obvious possible remedy is to weight the smaller sums so that a learning algorithm more easily avoids approximate solutions that exploit their insignificance which can be seen to be equivalent to repeating examples of the observations in minority classes. This can be achieved by naive random oversampling which is offered by this package along with other more advanced oversampling methods that function by generating synthetic data, which ideally would be analogous to one of the most plausible solutions to the class imbalance problem: collecting more data.

To our knowledge, there are no existing maintained Julia packages that implement resampling algorithms for multi-class classification problems or that handle both nominal and continuous features. This has served as a primary motivation for the creation of this package.
