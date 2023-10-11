

# SMOTE on Customer Churn Data


```julia

import Pkg;
Pkg.add(["Random", "CSV", "DataFrames", "MLJ", "Imbalance", "MLJBalancing", 
         "ScientificTypes","Impute", "StatsBase",  "Plots", "Measures", "HTTP"])

using Imbalance
using MLJBalancing
using CSV
using DataFrames
using ScientificTypes
using CategoricalArrays
using MLJ
using Plots
using Random
using HTTP: download
```

## Loading Data
In this example, we will consider the [Churn for Bank Customers](https://www.kaggle.com/datasets/mathchi/churn-for-bank-customers) found on Kaggle where the objective is to predict whether a customer is likely to leave a bank given financial and demographic features.

`CSV` gives us the ability to easily read the dataset after it's downloaded as follows


```julia
download("https://raw.githubusercontent.com/JuliaAI/Imbalance.jl/dev/docs/src/examples/smote_churn_dataset/churn.csv", "./")
df = CSV.read("./churn.csv", DataFrame)
first(df, 5) |> pretty
```

    ┌───────────┬────────────┬──────────┬─────────────┬───────────┬─────────┬───────┬────────┬────────────┬───────────────┬───────────┬────────────────┬─────────────────┬────────┐
    │ RowNumber │ CustomerId │ Surname  │ CreditScore │ Geography │ Gender  │ Age   │ Tenure │ Balance    │ NumOfProducts │ HasCrCard │ IsActiveMember │ EstimatedSalary │ Exited │
    │ Int64     │ Int64      │ String31 │ Int64       │ String7   │ String7 │ Int64 │ Int64  │ Float64    │ Int64         │ Int64     │ Int64          │ Float64         │ Int64  │
    │ Count     │ Count      │ Textual  │ Count       │ Textual   │ Textual │ Count │ Count  │ Continuous │ Count         │ Count     │ Count          │ Continuous      │ Count  │
    ├───────────┼────────────┼──────────┼─────────────┼───────────┼─────────┼───────┼────────┼────────────┼───────────────┼───────────┼────────────────┼─────────────────┼────────┤
    │ 1         │ 15634602   │ Hargrave │ 619         │ France    │ Female  │ 42    │ 2      │ 0.0        │ 1             │ 1         │ 1              │ 1.01349e5       │ 1      │
    │ 2         │ 15647311   │ Hill     │ 608         │ Spain     │ Female  │ 41    │ 1      │ 83807.9    │ 1             │ 0         │ 1              │ 1.12543e5       │ 0      │
    │ 3         │ 15619304   │ Onio     │ 502         │ France    │ Female  │ 42    │ 8      │ 1.59661e5  │ 3             │ 1         │ 0              │ 1.13932e5       │ 1      │
    │ 4         │ 15701354   │ Boni     │ 699         │ France    │ Female  │ 39    │ 1      │ 0.0        │ 2             │ 0         │ 0              │ 93826.6         │ 0      │
    │ 5         │ 15737888   │ Mitchell │ 850         │ Spain     │ Female  │ 43    │ 2      │ 1.25511e5  │ 1             │ 1         │ 1              │ 79084.1         │ 0      │
    └───────────┴────────────┴──────────┴─────────────┴───────────┴─────────┴───────┴────────┴────────────┴───────────────┴───────────┴────────────────┴─────────────────┴────────┘


There are plenty of useless columns that we can get rid of such as `RowNumber` and `CustomerID`. We also have to get rid of the categoircal features because SMOTE won't be able to deal with those; however, other variants such as SMOTE-NC can which we will consider in another tutorial.


```julia
df = df[:, Not([:RowNumber, :CustomerId, :Surname, 
           :Geography, :Gender])]

first(df, 5) |> pretty
```

    ┌─────────────┬───────┬────────┬────────────┬───────────────┬───────────┬────────────────┬─────────────────┬────────┐
    │ CreditScore │ Age   │ Tenure │ Balance    │ NumOfProducts │ HasCrCard │ IsActiveMember │ EstimatedSalary │ Exited │
    │ Int64       │ Int64 │ Int64  │ Float64    │ Int64         │ Int64     │ Int64          │ Float64         │ Int64  │
    │ Count       │ Count │ Count  │ Continuous │ Count         │ Count     │ Count          │ Continuous      │ Count  │
    ├─────────────┼───────┼────────┼────────────┼───────────────┼───────────┼────────────────┼─────────────────┼────────┤
    │ 619.0       │ 42.0  │ 2.0    │ 0.0        │ 1.0           │ 1.0       │ 1.0            │ 1.01349e5       │ 1.0    │
    │ 608.0       │ 41.0  │ 1.0    │ 83807.9    │ 1.0           │ 0.0       │ 1.0            │ 1.12543e5       │ 0.0    │
    │ 502.0       │ 42.0  │ 8.0    │ 1.59661e5  │ 3.0           │ 1.0       │ 0.0            │ 1.13932e5       │ 1.0    │
    │ 699.0       │ 39.0  │ 1.0    │ 0.0        │ 2.0           │ 0.0       │ 0.0            │ 93826.6         │ 0.0    │
    │ 850.0       │ 43.0  │ 2.0    │ 1.25511e5  │ 1.0           │ 1.0       │ 1.0            │ 79084.1         │ 0.0    │
    └─────────────┴───────┴────────┴────────────┴───────────────┴───────────┴────────────────┴─────────────────┴────────┘


Ideally, we may even remove ordinal variables because SMOTE will treat them as continuous and the synthetic data it generates will taking floating point values which will not occur in future data. Some models may be robust to this whatsoever and the main purpose of this tutorial is to later compare SMOTE-NC with SMOTE.

## Coercing Data

Let's coerce everything to continuous except for the target variable.


```julia
df = coerce(df, :Age=>Continuous,
                :Tenure=>Continuous,
                :Balance=>Continuous,
                :NumOfProducts=>Continuous,
                :HasCrCard=>Continuous,
                :IsActiveMember=>Continuous,
                :EstimatedSalary=>Continuous,
                :Exited=>Multiclass)

ScientificTypes.schema(df)
```


    ┌─────────────────┬───────────────┬─────────────────────────────────┐
    │ names           │ scitypes      │ types                           │
    ├─────────────────┼───────────────┼─────────────────────────────────┤
    │ CreditScore     │ Count         │ Int64                           │
    │ Age             │ Continuous    │ Float64                         │
    │ Tenure          │ Continuous    │ Float64                         │
    │ Balance         │ Continuous    │ Float64                         │
    │ NumOfProducts   │ Continuous    │ Float64                         │
    │ HasCrCard       │ Continuous    │ Float64                         │
    │ IsActiveMember  │ Continuous    │ Float64                         │
    │ EstimatedSalary │ Continuous    │ Float64                         │
    │ Exited          │ Multiclass{2} │ CategoricalValue{Int64, UInt32} │
    └─────────────────┴───────────────┴─────────────────────────────────┘



## Unpacking and Splitting Data

Both `MLJ` and the pure functional interface of `Imbalance` assume that the observations table `X` and target vector `y` are separate. We can accomplish that by using `unpack` from `MLJ`


```julia
y, X = unpack(df, ==(:Exited); rng=123);
first(X, 5) |> pretty
```

    ┌─────────────┬────────────┬────────────┬────────────┬───────────────┬────────────┬────────────────┬─────────────────┐
    │ CreditScore │ Age        │ Tenure     │ Balance    │ NumOfProducts │ HasCrCard  │ IsActiveMember │ EstimatedSalary │
    │ Int64       │ Float64    │ Float64    │ Float64    │ Float64       │ Float64    │ Float64        │ Float64         │
    │ Count       │ Continuous │ Continuous │ Continuous │ Continuous    │ Continuous │ Continuous     │ Continuous      │
    ├─────────────┼────────────┼────────────┼────────────┼───────────────┼────────────┼────────────────┼─────────────────┤
    │ 669.0       │ 31.0       │ 6.0        │ 1.13001e5  │ 1.0           │ 1.0        │ 0.0            │ 40467.8         │
    │ 822.0       │ 37.0       │ 3.0        │ 105563.0   │ 1.0           │ 1.0        │ 0.0            │ 1.82625e5       │
    │ 423.0       │ 36.0       │ 5.0        │ 97665.6    │ 1.0           │ 1.0        │ 0.0            │ 1.18373e5       │
    │ 623.0       │ 21.0       │ 10.0       │ 0.0        │ 2.0           │ 0.0        │ 1.0            │ 1.35851e5       │
    │ 691.0       │ 37.0       │ 7.0        │ 1.23068e5  │ 1.0           │ 1.0        │ 1.0            │ 98162.4         │
    └─────────────┴────────────┴────────────┴────────────┴───────────────┴────────────┴────────────────┴─────────────────┘


Splitting the data into train and test portions is also easy using `MLJ`'s `partition` function.


```julia
train_inds, test_inds = partition(eachindex(y), 0.8, shuffle=true, rng=Random.Xoshiro(42))
X_train, X_test = X[train_inds, :], X[test_inds, :]
y_train, y_test = y[train_inds], y[test_inds]
```


    (CategoricalValue{Int64, UInt32}[0, 1, 1, 0, 0, 0, 0, 0, 0, 0  …  0, 0, 0, 1, 0, 0, 0, 0, 1, 0], CategoricalValue{Int64, UInt32}[0, 0, 0, 0, 0, 1, 1, 0, 0, 0  …  0, 0, 0, 0, 0, 0, 0, 0, 0, 0])


## Oversampling



Before deciding to oversample, let's see how adverse is the imbalance problem, if it exists. Ideally, you may as well check if the classification model is robust to this problem.


```julia
checkbalance(y)         # comes from Imbalance
```

    1: ▇▇▇▇▇▇▇▇▇▇▇▇▇ 2037 (25.6%) 
    0: ▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇ 7963 (100.0%) 


Looks like we have a class imbalance problem. Let's oversample with SMOTE and set the desired ratios so that the positive minority class is 90% of the majority class


```julia
Xover, yover = smote(X_train, y_train; k=3, ratios=Dict(1=>0.9), rng=42)
checkbalance(yover)
```

    1: ▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇ 5736 (90.0%) 
    0: ▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇ 6373 (100.0%) 


## Training the Model



Because we have scientific types setup, we can easily check what models will be able to train on our data. This should guarantee that the model we choose won't throw an error due to types after feeding it the data.


```julia
models(matching(Xover, yover))
```


    54-element Vector{NamedTuple{(:name, :package_name, :is_supervised, :abstract_type, :deep_properties, :docstring, :fit_data_scitype, :human_name, :hyperparameter_ranges, :hyperparameter_types, :hyperparameters, :implemented_methods, :inverse_transform_scitype, :is_pure_julia, :is_wrapper, :iteration_parameter, :load_path, :package_license, :package_url, :package_uuid, :predict_scitype, :prediction_type, :reporting_operations, :reports_feature_importances, :supports_class_weights, :supports_online, :supports_training_losses, :supports_weights, :transform_scitype, :input_scitype, :target_scitype, :output_scitype)}}:
     (name = AdaBoostClassifier, package_name = MLJScikitLearnInterface, ... )
     (name = AdaBoostStumpClassifier, package_name = DecisionTree, ... )
     (name = BaggingClassifier, package_name = MLJScikitLearnInterface, ... )
     (name = BayesianLDA, package_name = MLJScikitLearnInterface, ... )
     (name = BayesianLDA, package_name = MultivariateStats, ... )
     (name = BayesianQDA, package_name = MLJScikitLearnInterface, ... )
     (name = BayesianSubspaceLDA, package_name = MultivariateStats, ... )
     (name = CatBoostClassifier, package_name = CatBoost, ... )
     (name = ConstantClassifier, package_name = MLJModels, ... )
     (name = DecisionTreeClassifier, package_name = BetaML, ... )
     ⋮
     (name = SGDClassifier, package_name = MLJScikitLearnInterface, ... )
     (name = SVC, package_name = LIBSVM, ... )
     (name = SVMClassifier, package_name = MLJScikitLearnInterface, ... )
     (name = SVMLinearClassifier, package_name = MLJScikitLearnInterface, ... )
     (name = SVMNuClassifier, package_name = MLJScikitLearnInterface, ... )
     (name = StableForestClassifier, package_name = SIRUS, ... )
     (name = StableRulesClassifier, package_name = SIRUS, ... )
     (name = SubspaceLDA, package_name = MultivariateStats, ... )
     (name = XGBoostClassifier, package_name = XGBoost, ... )


Let's go for a logistic classifier form MLJLinearModels


```julia
import Pkg; Pkg.add("MLJLinearModels")
```

### Before Oversampling


```julia
# 1. Load the model
LogisticClassifier = @load LogisticClassifier pkg=MLJLinearModels verbosity=0

# 2. Instantiate it
model = LogisticClassifier()

# 3. Wrap it with the data in a machine
mach = machine(model, X_train, y_train, scitype_check_level=0)

# 4. fit the machine learning model
fit!(mach, verbosity=0)
```


    trained Machine; caches model-specific representations of data
      model: LogisticClassifier(lambda = 2.220446049250313e-16, …)
      args: 
        1:	Source @113 ⏎ Table{Union{AbstractVector{Continuous}, AbstractVector{Count}}}
        2:	Source @972 ⏎ AbstractVector{Multiclass{2}}



### After Oversampling


```julia
# 3. Wrap it with the data in a machine
mach_over = machine(model, Xover, yover)

# 4. fit the machine learning model
fit!(mach_over)
```

## Evaluating the Model



To evaluate the model, we will use the balanced accuracy metric which equally account for all classes. 

### Before Oversampling


```julia
y_pred = predict_mode(mach, X_test)                         

score = round(balanced_accuracy(y_pred, y_test), digits=2)
```


    0.5


### After Oversampling


```julia
y_pred_over = predict_mode(mach_over, X_test)

score = round(balanced_accuracy(y_pred_over, y_test), digits=2)
```


    0.57


## Evaluating the Model - Revisited

We have previously evaluated the model using a single point estimate of the balanced accuracy resulting in a `7%` improvement. A more precise evaluation would use cross validation to combine many different point estimates into a more precise one (their average). The standard deviation among such point estimates also allows us to quantify the uncertainty of the estimate; a smaller standard deviation would imply a smaller confidence interval at the same probability.

### Before Oversampling


```julia
cv=CV(nfolds=10)
evaluate!(mach, resampling=cv, measure=balanced_accuracy) 
```

    Evaluating over 10 folds: 100%[=========================] Time: 0:00:00[K



    PerformanceEvaluation object with these fields:
      model, measure, operation, measurement, per_fold,
      per_observation, fitted_params_per_fold,
      report_per_fold, train_test_rows, resampling, repeats
    Extract:
    ┌─────────────────────┬──────────────┬─────────────┬──────────┬─────────────────
    │ measure             │ operation    │ measurement │ 1.96*SE  │ per_fold       ⋯
    ├─────────────────────┼──────────────┼─────────────┼──────────┼─────────────────
    │ BalancedAccuracy(   │ predict_mode │ 0.5         │ 3.29e-16 │ [0.5, 0.5, 0.5 ⋯
    │   adjusted = false) │              │             │          │                ⋯
    └─────────────────────┴──────────────┴─────────────┴──────────┴─────────────────
                                                                    1 column omitted



This looks good. Negligble standard deviation; point estimates are all centered around `0.5`.

### After Oversampling

At first glance, this seems really nontrivial since resampling will have to be performed before training the model on each fold during cross-validation. Thankfully, the `MLJBalancing` helps us avoid doing this manually by offering `BalancedModel` where we can wrap any `MLJ` classification model with an arbitrary number of `Imbalance.jl` resamplers in a pipeline that behaves like a single `MLJ` model.

In this, we must construct the resampling model via it's `MLJ` interface then pass it along with the classification model to `BalancedModel`.


```julia
# 2. Instantiate the models
oversampler = Imbalance.MLJ.SMOTE(k=3, ratios=Dict(1=>0.9), rng=42)

# 2.1 Wrap them in one model
balanced_model = BalancedModel(model=model, balancer1=oversampler)

# 3. Wrap it with the data in a machine
mach_over = machine(balanced_model, X_train, y_train, scitype_check_level=0)

# 4. fit the machine learning model
fit!(mach_over, verbosity=0)
```


    trained Machine; does not cache data
      model: BalancedModelProbabilistic(model = LogisticClassifier(lambda = 2.220446049250313e-16, …), …)
      args: 
        1:	Source @991 ⏎ Table{Union{AbstractVector{Continuous}, AbstractVector{Count}}}
        2:	Source @939 ⏎ AbstractVector{Multiclass{2}}



We can easily confirm that this is equivalent to what we had earlier


```julia
predict_mode(mach_over, X_test) == y_pred_over
```


    true


Now let's cross-validate


```julia
cv=CV(nfolds=10)
evaluate!(mach_over, resampling=cv, measure=balanced_accuracy) 
```

    Evaluating over 10 folds: 100%[=========================] Time: 0:00:00[K



    PerformanceEvaluation object with these fields:
      model, measure, operation, measurement, per_fold,
      per_observation, fitted_params_per_fold,
      report_per_fold, train_test_rows, resampling, repeats
    Extract:
    ┌─────────────────────┬──────────────┬─────────────┬─────────┬──────────────────
    │ measure             │ operation    │ measurement │ 1.96*SE │ per_fold        ⋯
    ├─────────────────────┼──────────────┼─────────────┼─────────┼──────────────────
    │ BalancedAccuracy(   │ predict_mode │ 0.552       │ 0.0145  │ [0.549, 0.563,  ⋯
    │   adjusted = false) │              │             │         │                 ⋯
    └─────────────────────┴──────────────┴─────────────┴─────────┴──────────────────
                                                                    1 column omitted



The improvement is about `5.2%` after cross-validation. If we are further to assume scores to be normally distributed, then the `95%` confidence interval is `5.2±1.45%` improvement. Let's see if this gets any better when we rather use `SMOTE-NC` in a later example.



