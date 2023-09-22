```julia
using Imbalance
using CSV
using DataFrames
using ScientificTypes
using CategoricalArrays
using MLJ
using Plots
using Random
```

## Loading Data
In this example, we will consider the [Churn for Bank Customers](https://www.kaggle.com/datasets/mathchi/churn-for-bank-customers) found on Kaggle where the objective is to predict whether a customer is likely to leave a bank given financial and demographic features. 

We already considered this dataset using SMOTE, in this example we see if the results are any better using SMOTE-NC.


```julia
df = CSV.read("datasets/churn.csv", DataFrame)
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
checkbalance(y)
```

    1: ▇▇▇▇▇▇▇▇▇▇▇▇▇ 2037 (25.6%) 
    0: ▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇ 7963 (100.0%) 


Looks like we have a class imbalance problem. Let's oversample with SMOTE-NC and set the desired ratios so that the positive minority class is 90% of the majority class


```julia
Xover, yover = smotenc(X, y; k=3, ratios=Dict(1=>0.9), rng=42)
```


```julia
checkbalance(yover)
```

    1: ▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇ 7167 (90.0%) 
    0: ▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇ 7963 (100.0%) 


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


Let's go for a logistic classifier form MLJLinearModels


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
    └ @ Main /Users/essam/.julia/packages/MLJModels/7apZ3/src/loading.jl:159



    trained Machine; caches model-specific representations of data
      model: DecisionTreeClassifier(max_depth = 4, …)
      args: 
        1:	Source @966 ⏎ Table{Union{AbstractVector{Continuous}, AbstractVector{Multiclass{3}}, AbstractVector{Multiclass{2}}, AbstractVector{OrderedFactor{460}}, AbstractVector{OrderedFactor{70}}, AbstractVector{OrderedFactor{11}}, AbstractVector{OrderedFactor{4}}}}
        2:	Source @230 ⏎ AbstractVector{Multiclass{2}}



### After Oversampling


```julia
# 3. Wrap it with the data in a machine
mach_over = machine(model, Xover, yover)

# 4. fit the machine learning model
fit!(mach_over)
```

    ┌ Info: Training machine(DecisionTreeClassifier(max_depth = 4, …), …).
    └ @ MLJBase /Users/essam/.julia/packages/MLJBase/0rn2V/src/machines.jl:492



    trained Machine; caches model-specific representations of data
      model: DecisionTreeClassifier(max_depth = 4, …)
      args: 
        1:	Source @511 ⏎ Table{Union{AbstractVector{Continuous}, AbstractVector{Multiclass{3}}, AbstractVector{Multiclass{2}}, AbstractVector{OrderedFactor{460}}, AbstractVector{OrderedFactor{70}}, AbstractVector{OrderedFactor{11}}, AbstractVector{OrderedFactor{4}}}}
        2:	Source @112 ⏎ AbstractVector{Multiclass{2}}



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


Although the results do get better compared to when we just used SMOTE, it holds in this case that the extra categorical features we took into account aren't that important. The difference can be attributed to the decision tree.



