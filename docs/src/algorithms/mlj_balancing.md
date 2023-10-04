# Combining Resamplers

Resampling methods can be combined sequentially or in parallel, along with a classification model, to yield hybrid or ensemble models that may be even more powerful than using the classification model with only one of the individual resamplers.

## Sequential Resampling

`MLJBalancing.jl` allows chaining an arbitrary number of resamplers from `Imbalance.jl` (also called *balancers*) with classification models from `MLJ` via `BalancedModel`. This makes it possible to use `BalancedModel` to form hybrid resampling methods that combine oversampling and under-sampling methods in a linear pipeline such as `SMOTE-Tomek` and `SMOTE-ENN`.


#### Construct the resampler and classification models
```julia
SMOTE = @load SMOTE pkg=Imbalance verbosity=0
TomekUndersampler = @load TomekUndersampler pkg=Imbalance verbosity=0
LogisticClassifier = @load LogisticClassifier pkg=MLJLinearModels verbosity=0

oversampler = SMOTE(k=5, ratios=1.0, rng=42)
undersampler = TomekUndersampler(min_ratios=0.5, rng=42)

logistic_model = LogisticClassifier()
```

#### Wrap them all in BalancedModel
```julia
balanced_model = BalancedModel(model=logistic_model, 
                               balancer1=oversampler, balancer2=undersampler)
```
Here training data will be passed to `balancer1` then `balancer2`, whose output is used to train the classifier `model`.  In prediction, the resamplers `balancer1` and `blancer2` are bypassed and in general. At this point, they behave like one single `MLJ` model that can be fit, validated or fine-tuned like any other.

In general, there can be any number of balancers, and the user can give the balancers arbitrary names. 


## Parallel Resampling with Balanced Bagging

`MLJBalancing.jl` also offers an implementation of bagging over probabilistic classifiers where the majority class is randomly undersampled `T` times down to the size of the minority class then a model is trained on each of the `T` undersampled datasets. The predictions are then aggregated by averaging. This is offered via `BalancedBaggingClassifier` and can be only used for binary classification.

```julia
BalancedBaggingClassifier(model=nothing, T=0, rng = Random.default_rng(),)
```
#### Arguments

- `model::Probabilistic`: A probabilistic classification model that implements the `MLJModelInterface`

- `T::Integer=0`: The number of bags to be used in the ensemble. If not given, will be set as
    the ratio between the frequency of the majority and minority classes.

- `rng::Union{AbstractRNG, Integer}=default_rng()`: Either an `AbstractRNG` object or an `Integer` 
seed to be used with `Xoshiro`

#### Example
```julia
using MLJ
using Imbalance
using MLJBalancing

X, y = generate_imbalanced_data(100, 5; cat_feats_num_vals = [3, 2], 
                                        probs = [0.9, 0.1], 
                                        type = "ColTable", 
                                        rng=42)

LogisticClassifier = @load LogisticClassifier pkg=MLJLinearModels verbosity=0
logistic_model = LogisticClassifier()
bagging_model = BalancedBaggingClassifier(model=logistic_model, T=10, rng=Random.Xoshiro(42))
```
Now you can fit, predict, cross-validate and finetune it like any other probabilistic MLJ model where `X` must be a table input (e.g., a dataframe).
```julia
mach = machine(bagging_model, X, y)
fit!(mach)
pred = predict(mach, X)
```
