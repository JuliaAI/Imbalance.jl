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

`CSV` gives us the ability to easily read the dataset after it's downloaded as follows


```julia
df = CSV.read("../datasets/churn.csv", DataFrame)
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


There are plenty of useless columns that we can get rid of such as `RowNumber` and `CustomerID`. We also have to get rid of the cateogircal features because SMOTE won't be able to deal with those; however, other variants such as SMOTE-NC can which we will consider in another tutorial.


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
Xover, yover = smote(X, y; k=3, ratios=Dict(1=>0.9), rng=42)
checkbalance(yover)
```

    1: ▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇ 7167 (90.0%) 
    0: ▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇ 7963 (100.0%) 


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

        Updating registry at `~/.julia/registries/General.toml`
    ┌ Error: Some registries failed to update:
    │     — /Users/essam/.julia/registries/General.toml — failed to download from https://pkg.julialang.org/registry/23338594-aafe-5451-b93e-139f81909106/95646b6cd2d61c2d6784757067e14d5bcb846090. Exception: HTTP/2 200 (Operation too slow. Less than 1 bytes/sec transferred the last 20 seconds) while requesting https://pkg.julialang.org/registry/23338594-aafe-5451-b93e-139f81909106/95646b6cd2d61c2d6784757067e14d5bcb846090
    └ @ Pkg.Registry /Users/julia/.julia/scratchspaces/a66863c6-20e8-4ff4-8a62-49f30b1f605e/agent-cache/default-macmini-aarch64-4.0/build/default-macmini-aarch64-4-0/julialang/julia-release-1-dot-8/usr/share/julia/stdlib/v1.8/Pkg/src/Registry/Registry.jl:449
       Resolving package versions...
        Updating `~/Documents/GitHub/Imbalance.jl/Project.toml`
      [6ee0df7b] + MLJLinearModels v0.9.2
        Updating `~/Documents/GitHub/Imbalance.jl/Manifest.toml`
      [6a86dc24] + FiniteDiff v2.21.1
      [42fd0dbc] + IterativeSolvers v0.9.2
      [d3d80556] + LineSearches v7.2.0
      [7a12625a] + LinearMaps v3.11.0
      [6ee0df7b] + MLJLinearModels v0.9.2
      [d41bc354] + NLSolversBase v7.8.3
      [429524aa] + Optim v1.7.7
      [85a6dd25] + PositiveFactorizations v0.2.4
      [3cdcf5f2] + RecipesBase v1.3.4


### Before Oversampling


```julia
# 1. Load the model
LogisticClassifier = @load LogisticClassifier pkg=MLJLinearModels verbosity=0

# 2. Instantiate it
model = LogisticClassifier()

# 3. Wrap it with the data in a machine
mach = machine(model, X_train, y_train)

# 4. fit the machine learning model
fit!(mach, verbosity=0)
```

    ┌ Warning: The number and/or types of data arguments do not match what the specified model
    │ supports. Suppress this type check by specifying `scitype_check_level=0`.
    │ 
    │ Run `@doc MLJLinearModels.LogisticClassifier` to learn more about your model's requirements.
    │ 
    │ Commonly, but non exclusively, supervised models are constructed using the syntax
    │ `machine(model, X, y)` or `machine(model, X, y, w)` while most other models are
    │ constructed with `machine(model, X)`.  Here `X` are features, `y` a target, and `w`
    │ sample or class weights.
    │ 
    │ In general, data in `machine(model, data...)` is expected to satisfy
    │ 
    │     scitype(data) <: MLJ.fit_data_scitype(model)
    │ 
    │ In the present case:
    │ 
    │ scitype(data) = Tuple{Table{Union{AbstractVector{Continuous}, AbstractVector{Count}}}, AbstractVector{Multiclass{2}}}
    │ 
    │ fit_data_scitype(model) = Tuple{Table{<:AbstractVector{<:Continuous}}, AbstractVector{<:Finite}}
    └ @ MLJBase /Users/essam/.julia/packages/MLJBase/ByFwA/src/machines.jl:230



    trained Machine; caches model-specific representations of data
      model: LogisticClassifier(lambda = 2.220446049250313e-16, …)
      args: 
        1:	Source @148 ⏎ Table{Union{AbstractVector{Continuous}, AbstractVector{Count}}}
        2:	Source @042 ⏎ AbstractVector{Multiclass{2}}



### After Oversampling


```julia
# 3. Wrap it with the data in a machine
mach_over = machine(model, Xover, yover)

# 4. fit the machine learning model
fit!(mach_over)
```

    ┌ Info: Training machine(LogisticClassifier(lambda = 2.220446049250313e-16, …), …).
    └ @ MLJBase /Users/essam/.julia/packages/MLJBase/ByFwA/src/machines.jl:492
    ┌ Info: Solver: MLJLinearModels.LBFGS{Optim.Options{Float64, Nothing}, NamedTuple{(), Tuple{}}}
    │   optim_options: Optim.Options{Float64, Nothing}
    │   lbfgs_options: NamedTuple{(), Tuple{}} NamedTuple()
    └ @ MLJLinearModels /Users/essam/.julia/packages/MLJLinearModels/zSQnL/src/mlj/interface.jl:72



    trained Machine; caches model-specific representations of data
      model: LogisticClassifier(lambda = 2.220446049250313e-16, …)
      args: 
        1:	Source @525 ⏎ Table{AbstractVector{Continuous}}
        2:	Source @636 ⏎ AbstractVector{Multiclass{2}}



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


    0.66




