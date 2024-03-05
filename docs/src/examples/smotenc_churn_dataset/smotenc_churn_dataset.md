

# SMOTENC on Customer Churn Data


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

We already considered this dataset using SMOTE, in this example we see if the results are any better using SMOTE-NC.


```julia
download("https://raw.githubusercontent.com/JuliaAI/Imbalance.jl/dev/docs/src/examples/smotenc_churn_dataset/churn.csv", "./")
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


Let's get rid of useless columns such as `RowNumber` and `CustomerId`


```julia
df = df[:, Not([:Surname, :RowNumber, :CustomerId])]

first(df, 5) |> pretty
```

    ┌─────────────┬───────────┬─────────┬───────┬────────┬────────────┬───────────────┬───────────┬────────────────┬─────────────────┬────────┐
    │ CreditScore │ Geography │ Gender  │ Age   │ Tenure │ Balance    │ NumOfProducts │ HasCrCard │ IsActiveMember │ EstimatedSalary │ Exited │
    │ Int64       │ String7   │ String7 │ Int64 │ Int64  │ Float64    │ Int64         │ Int64     │ Int64          │ Float64         │ Int64  │
    │ Count       │ Textual   │ Textual │ Count │ Count  │ Continuous │ Count         │ Count     │ Count          │ Continuous      │ Count  │
    ├─────────────┼───────────┼─────────┼───────┼────────┼────────────┼───────────────┼───────────┼────────────────┼─────────────────┼────────┤
    │ 619         │ France    │ Female  │ 42    │ 2      │ 0.0        │ 1             │ 1         │ 1              │ 1.01349e5       │ 1      │
    │ 608         │ Spain     │ Female  │ 41    │ 1      │ 83807.9    │ 1             │ 0         │ 1              │ 1.12543e5       │ 0      │
    │ 502         │ France    │ Female  │ 42    │ 8      │ 1.59661e5  │ 3             │ 1         │ 0              │ 1.13932e5       │ 1      │
    │ 699         │ France    │ Female  │ 39    │ 1      │ 0.0        │ 2             │ 0         │ 0              │ 93826.6         │ 0      │
    │ 850         │ Spain     │ Female  │ 43    │ 2      │ 1.25511e5  │ 1             │ 1         │ 1              │ 79084.1         │ 0      │
    └─────────────┴───────────┴─────────┴───────┴────────┴────────────┴───────────────┴───────────┴────────────────┴─────────────────┴────────┘


## Coercing Data

Let's coerce the nominal data to `Multiclass`, the ordinal data to `OrderedFactor` and the continuous data to `Continuous`.


```julia
df = coerce(df, 
              :Geography => Multiclass, 
              :Gender=> Multiclass,
              :CreditScore => OrderedFactor,
              :Age => OrderedFactor,
              :Tenure => OrderedFactor,
              :Balance => Continuous,
              :NumOfProducts => OrderedFactor,
              :HasCrCard => Multiclass,
              :IsActiveMember => Multiclass,
              :EstimatedSalary => Continuous,
              :Exited => Multiclass
              )

ScientificTypes.schema(df)
```


    ┌─────────────────┬────────────────────┬───────────────────────────────────┐
    │ names           │ scitypes           │ types                             │
    ├─────────────────┼────────────────────┼───────────────────────────────────┤
    │ CreditScore     │ OrderedFactor{460} │ CategoricalValue{Int64, UInt32}   │
    │ Geography       │ Multiclass{3}      │ CategoricalValue{String7, UInt32} │
    │ Gender          │ Multiclass{2}      │ CategoricalValue{String7, UInt32} │
    │ Age             │ OrderedFactor{70}  │ CategoricalValue{Int64, UInt32}   │
    │ Tenure          │ OrderedFactor{11}  │ CategoricalValue{Int64, UInt32}   │
    │ Balance         │ Continuous         │ Float64                           │
    │ NumOfProducts   │ OrderedFactor{4}   │ CategoricalValue{Int64, UInt32}   │
    │ HasCrCard       │ Multiclass{2}      │ CategoricalValue{Int64, UInt32}   │
    │ IsActiveMember  │ Multiclass{2}      │ CategoricalValue{Int64, UInt32}   │
    │ EstimatedSalary │ Continuous         │ Float64                           │
    │ Exited          │ Multiclass{2}      │ CategoricalValue{Int64, UInt32}   │
    └─────────────────┴────────────────────┴───────────────────────────────────┘



## Unpacking and Splitting Data


```julia
y, X = unpack(df, ==(:Exited); rng=123);
first(X, 5) |> pretty
```

    ┌─────────────────────────────────┬───────────────────────────────────┬───────────────────────────────────┬─────────────────────────────────┬─────────────────────────────────┬────────────┬─────────────────────────────────┬─────────────────────────────────┬─────────────────────────────────┬─────────────────┐
    │ CreditScore                     │ Geography                         │ Gender                            │ Age                             │ Tenure                          │ Balance    │ NumOfProducts                   │ HasCrCard                       │ IsActiveMember                  │ EstimatedSalary │
    │ CategoricalValue{Int64, UInt32} │ CategoricalValue{String7, UInt32} │ CategoricalValue{String7, UInt32} │ CategoricalValue{Int64, UInt32} │ CategoricalValue{Int64, UInt32} │ Float64    │ CategoricalValue{Int64, UInt32} │ CategoricalValue{Int64, UInt32} │ CategoricalValue{Int64, UInt32} │ Float64         │
    │ OrderedFactor{460}              │ Multiclass{3}                     │ Multiclass{2}                     │ OrderedFactor{70}               │ OrderedFactor{11}               │ Continuous │ OrderedFactor{4}                │ Multiclass{2}                   │ Multiclass{2}                   │ Continuous      │
    ├─────────────────────────────────┼───────────────────────────────────┼───────────────────────────────────┼─────────────────────────────────┼─────────────────────────────────┼────────────┼─────────────────────────────────┼─────────────────────────────────┼─────────────────────────────────┼─────────────────┤
    │ 669                             │ France                            │ Female                            │ 31                              │ 6                               │ 1.13001e5  │ 1                               │ 1                               │ 0                               │ 40467.8         │
    │ 822                             │ France                            │ Male                              │ 37                              │ 3                               │ 105563.0   │ 1                               │ 1                               │ 0                               │ 1.82625e5       │
    │ 423                             │ France                            │ Female                            │ 36                              │ 5                               │ 97665.6    │ 1                               │ 1                               │ 0                               │ 1.18373e5       │
    │ 623                             │ France                            │ Male                              │ 21                              │ 10                              │ 0.0        │ 2                               │ 0                               │ 1                               │ 1.35851e5       │
    │ 691                             │ Germany                           │ Female                            │ 37                              │ 7                               │ 1.23068e5  │ 1                               │ 1                               │ 1                               │ 98162.4         │
    └─────────────────────────────────┴───────────────────────────────────┴───────────────────────────────────┴─────────────────────────────────┴─────────────────────────────────┴────────────┴─────────────────────────────────┴─────────────────────────────────┴─────────────────────────────────┴─────────────────┘



```julia
train_inds, test_inds = partition(eachindex(y), 0.8, shuffle=true, 
                                  rng=Random.Xoshiro(42))
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


Looks like we have a class imbalance problem. Let's oversample with SMOTE-NC and set the desired ratios so that the positive minority class is 90% of the majority class


```julia
Xover, yover = smotenc(X_train, y_train; k=3, ratios=Dict(1=>0.9), rng=42)
```


    (12109×10 DataFrame
       Row │ CreditScore  Geography  Gender  Age   Tenure  Balance         NumOfPr ⋯
           │ Cat…         Cat…       Cat…    Cat…  Cat…    Float64         Cat…    ⋯
    ───────┼────────────────────────────────────────────────────────────────────────
         1 │ 551          France     Female  38    10           0.0        2       ⋯
         2 │ 676          France     Female  37    5        89634.7        1
         3 │ 543          France     Male    42    4        89838.7        3
         4 │ 663          France     Male    34    10           0.0        1
         5 │ 621          Germany    Female  34    2        91258.5        2       ⋯
         6 │ 723          France     Male    28    4            0.0        2
         7 │ 735          France     Female  21    1            1.78718e5  2
         8 │ 501          France     Male    35    6        99760.8        1
       ⋮   │      ⋮           ⋮        ⋮      ⋮      ⋮           ⋮               ⋮ ⋱
     12103 │ 551          France     Female  40    2            1.68002e5  1       ⋯
     12104 │ 716          France     Female  46    2            1.09379e5  2
     12105 │ 850          Spain      Female  45    10           1.66777e5  1
     12106 │ 785          France     Female  39    9            1.33118e5  1
     12107 │ 565          Germany    Female  39    5            1.44874e5  1       ⋯
     12108 │ 510          Germany    Male    43    0            1.38862e5  1
     12109 │ 760          France     Female  41    2       113419.0        1
                                                    4 columns and 12094 rows omitted, CategoricalValue{Int64, UInt32}[0, 1, 1, 0, 0, 0, 0, 0, 0, 0  …  1, 1, 1, 1, 1, 1, 1, 1, 1, 1])



```julia
checkbalance(yover)
```

    1: ▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇ 5736 (90.0%) 
    0: ▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇ 6373 (100.0%) 


## Training the Model



Let's find possible models


```julia
ms = models(matching(Xover, yover))
```


    5-element Vector{NamedTuple{(:name, :package_name, :is_supervised, :abstract_type, :deep_properties, :docstring, :fit_data_scitype, :human_name, :hyperparameter_ranges, :hyperparameter_types, :hyperparameters, :implemented_methods, :inverse_transform_scitype, :is_pure_julia, :is_wrapper, :iteration_parameter, :load_path, :package_license, :package_url, :package_uuid, :predict_scitype, :prediction_type, :reporting_operations, :reports_feature_importances, :supports_class_weights, :supports_online, :supports_training_losses, :supports_weights, :transform_scitype, :input_scitype, :target_scitype, :output_scitype)}}:
     (name = CatBoostClassifier, package_name = CatBoost, ... )
     (name = ConstantClassifier, package_name = MLJModels, ... )
     (name = DecisionTreeClassifier, package_name = BetaML, ... )
     (name = DeterministicConstantClassifier, package_name = MLJModels, ... )
     (name = RandomForestClassifier, package_name = BetaML, ... )


Let's go for a decision tree classifier from [BetaML](https://github.com/sylvaticus/BetaML.jl).

```julia
import Pkg; Pkg.add("BetaML")
```

Let's go for a decision tree from BetaML. We can't go for logistic regression as we did in the SMOTE tutorial because it does not support categotical features.

### Before Oversampling


```julia
# 1. Load the model
DecisionTreeClassifier = @load DecisionTreeClassifier pkg=BetaML

# 2. Instantiate it
model = DecisionTreeClassifier( max_depth=4, rng=Random.Xoshiro(42))

# 3. Wrap it with the data in a machine
mach = machine(model, X_train, y_train)

# 4. fit the machine learning model
fit!(mach, verbosity=0)
```

    import BetaML ✔


    ┌ Info: For silent loading, specify `verbosity=0`. 
    └ @ Main /Users/essam/.julia/packages/MLJModels/EkXIe/src/loading.jl:159



    trained Machine; caches model-specific representations of data
      model: DecisionTreeClassifier(max_depth = 4, …)
      args: 
        1:	Source @378 ⏎ Table{Union{AbstractVector{Continuous}, AbstractVector{Multiclass{3}}, AbstractVector{Multiclass{2}}, AbstractVector{OrderedFactor{460}}, AbstractVector{OrderedFactor{70}}, AbstractVector{OrderedFactor{11}}, AbstractVector{OrderedFactor{4}}}}
        2:	Source @049 ⏎ AbstractVector{Multiclass{2}}



### After Oversampling


```julia
# 3. Wrap it with the data in a machine
mach_over = machine(model, Xover, yover)

# 4. fit the machine learning model
fit!(mach_over)
```

    ┌ Info: Training machine(DecisionTreeClassifier(max_depth = 4, …), …).
    └ @ MLJBase /Users/essam/.julia/packages/MLJBase/ByFwA/src/machines.jl:492



    trained Machine; caches model-specific representations of data
      model: DecisionTreeClassifier(max_depth = 4, …)
      args: 
        1:	Source @033 ⏎ Table{Union{AbstractVector{Continuous}, AbstractVector{Multiclass{3}}, AbstractVector{Multiclass{2}}, AbstractVector{OrderedFactor{460}}, AbstractVector{OrderedFactor{70}}, AbstractVector{OrderedFactor{11}}, AbstractVector{OrderedFactor{4}}}}
        2:	Source @939 ⏎ AbstractVector{Multiclass{2}}



## Evaluating the Model



To evaluate the model, we will use the balanced accuracy metric which equally accounts for all classes. 

### Before Oversampling


```julia
y_pred = predict_mode(mach, X_test)                         

score = round(balanced_accuracy(y_pred, y_test), digits=2)
```


    0.57


### After Oversampling


```julia
y_pred_over = predict_mode(mach_over, X_test)

score = round(balanced_accuracy(y_pred_over, y_test), digits=2)
```


    0.7


Although the results do get better compared to when we just used SMOTE, it may hold in this case that the extra categorical features we took into account are not be that important. The difference may be attributed to the decision tree.

## Evaluating the Model - Revisited

We have previously evaluated the model using a single point estimate of the balanced accuracy resulting in a `13%` improvement. A more precise evaluation would use cross validation to combine many different point estimates into a more precise one (their average). The standard deviation among such point estimates also allows us to quantify the uncertainty of the estimate; a smaller standard deviation would imply a smaller confidence interval at the same probability.

### Before Oversampling


```julia
cv=CV(nfolds=10)
evaluate!(mach, resampling=cv, measure=balanced_accuracy) 
```

    Evaluating over 10 folds: 100%[=========================] Time: 0:02:54[K



    PerformanceEvaluation object with these fields:
      model, measure, operation, measurement, per_fold,
      per_observation, fitted_params_per_fold,
      report_per_fold, train_test_rows, resampling, repeats
    Extract:
    ┌─────────────────────┬──────────────┬─────────────┬─────────┬──────────────────
    │ measure             │ operation    │ measurement │ 1.96*SE │ per_fold        ⋯
    ├─────────────────────┼──────────────┼─────────────┼─────────┼──────────────────
    │ BalancedAccuracy(   │ predict_mode │ 0.565       │ 0.00623 │ [0.568, 0.554,  ⋯
    │   adjusted = false) │              │             │         │                 ⋯
    └─────────────────────┴──────────────┴─────────────┴─────────┴──────────────────
                                                                    1 column omitted



Before oversampling, and assuming that the balanced accuracy score is normally distribued we can be `95%` confident that the balanced accuracy on new data is `56.5±0.62`. Indeed, this agrees a lot with the original point estimate.

### After Oversampling

At first glance, this seems really nontrivial since resampling will have to be performed before training the model on each fold during cross-validation. Thankfully, the `MLJBalancing` helps us avoid doing this manually by offering `BalancedModel` where we can wrap any `MLJ` classification model with an arbitrary number of `Imbalance.jl` resamplers in a pipeline that behaves like a single `MLJ` model.

In this, we must construct the resampling model via it's `MLJ` interface then pass it along with the classification model to `BalancedModel`.


```julia
# 2. Instantiate the models
oversampler = Imbalance.MLJ.SMOTENC(k=3, ratios=Dict(1=>0.9), rng=42)

# 2.1 Wrap them in one model
balanced_model = BalancedModel(model=model, balancer1=oversampler)

# 3. Wrap it with the data in a machine
mach_over = machine(balanced_model, X_train, y_train, scitype_check_level=0)

# 4. fit the machine learning model
fit!(mach_over, verbosity=0)
```


    trained Machine; does not cache data
      model: BalancedModelProbabilistic(model = DecisionTreeClassifier(max_depth = 4, …), …)
      args: 
        1:	Source @967 ⏎ Table{Union{AbstractVector{Continuous}, AbstractVector{Multiclass{3}}, AbstractVector{Multiclass{2}}, AbstractVector{OrderedFactor{460}}, AbstractVector{OrderedFactor{70}}, AbstractVector{OrderedFactor{11}}, AbstractVector{OrderedFactor{4}}}}
        2:	Source @394 ⏎ AbstractVector{Multiclass{2}}



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

    Evaluating over 10 folds: 100%[=========================] Time: 0:07:24[K



    PerformanceEvaluation object with these fields:
      model, measure, operation, measurement, per_fold,
      per_observation, fitted_params_per_fold,
      report_per_fold, train_test_rows, resampling, repeats
    Extract:
    ┌─────────────────────┬──────────────┬─────────────┬─────────┬──────────────────
    │ measure             │ operation    │ measurement │ 1.96*SE │ per_fold        ⋯
    ├─────────────────────┼──────────────┼─────────────┼─────────┼──────────────────
    │ BalancedAccuracy(   │ predict_mode │ 0.677       │ 0.0124  │ [0.678, 0.688,  ⋯
    │   adjusted = false) │              │             │         │                 ⋯
    └─────────────────────┴──────────────┴─────────────┴─────────┴──────────────────
                                                                    1 column omitted



Fair enough. After oversampling the interval under the same assumptions is `67.7±1.2%` which is still a meaningful improvement over `56.5±0.62` that we had prior to oversampling ot the `55.2±1.5%` that we had with logistic regression and SMOTE in an earlier example.



