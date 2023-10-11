

# SMOTE-Tomek for Ethereum Fraud Detection


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
using Impute
using HTTP: download
```

## Loading Data
In this example, we will consider the [Ethereum Fraud Detection Dataset](https://www.kaggle.com/datasets/vagifa/ethereum-frauddetection-dataset) found on Kaggle where the objective is to predict whether an Ethereum transaction is fraud or not (called `FLAG`) given some features about the transaction.

`CSV` gives us the ability to easily read the dataset after it's downloaded as follows


```julia
download("https://raw.githubusercontent.com/JuliaAI/Imbalance.jl/dev/docs/src/examples/fraud_detection/transactions.csv", "./")

df = CSV.read("./transactions.csv", DataFrame)
first(df, 5) |> pretty
```

There are plenty of useless columns that we can get rid of such as `Column1`, `Index` and probably, `Address`. We also have to get rid of the categorical features because SMOTE won't be able to deal with those and it leaves us with more options for the model.


```julia
df = df[:,
	Not([
		:Column1,
		:Index,
		:Address,
		Symbol(" ERC20 most sent token type"),
		Symbol(" ERC20_most_rec_token_type"),
	]),
] 
first(df, 5) |> pretty
```

If you scroll through the printed data frame, you find that some columns also have `Missing` for their element type, meaning that they may be containing missing values. We will use *linear interpolation*, *last-observation carried forward* and *next observation carried backward* techniques to fill up the missing values. This will allow us to call `disallowmissing!(df)` to return a dataframe where `Missing` is not an element type for any column.


```julia
df = Impute.interp(df) |> Impute.locf() |> Impute.nocb(); disallowmissing!(df)
first(df, 5) |> pretty
```

## Coercing Data

Let's look at the schema first


```julia
ScientificTypes.schema(df)
```


    ┌──────────────────────────────────────────────────────┬────────────┬─────────┐
    │ names                                                │ scitypes   │ types   │
    ├──────────────────────────────────────────────────────┼────────────┼─────────┤
    │ FLAG                                                 │ Count      │ Int64   │
    │ Avg min between sent tnx                             │ Continuous │ Float64 │
    │ Avg min between received tnx                         │ Continuous │ Float64 │
    │ Time Diff between first and last (Mins)              │ Continuous │ Float64 │
    │ Sent tnx                                             │ Count      │ Int64   │
    │ Received Tnx                                         │ Count      │ Int64   │
    │ Number of Created Contracts                          │ Count      │ Int64   │
    │ Unique Received From Addresses                       │ Count      │ Int64   │
    │ Unique Sent To Addresses                             │ Count      │ Int64   │
    │ min value received                                   │ Continuous │ Float64 │
    │ max value received                                   │ Continuous │ Float64 │
    │ avg val received                                     │ Continuous │ Float64 │
    │ min val sent                                         │ Continuous │ Float64 │
    │ max val sent                                         │ Continuous │ Float64 │
    │ avg val sent                                         │ Continuous │ Float64 │
    │ min value sent to contract                           │ Continuous │ Float64 │
    │                          ⋮                           │     ⋮      │    ⋮    │
    └──────────────────────────────────────────────────────┴────────────┴─────────┘
                                                                    30 rows omitted



The `FLAG` target should definitely be Multiclass, the rest seems fine.


```julia
df = coerce(df, :FLAG =>Multiclass)
ScientificTypes.schema(df)
```


    ┌──────────────────────────────────────────────────────┬───────────────┬────────
    │ names                                                │ scitypes      │ types ⋯
    ├──────────────────────────────────────────────────────┼───────────────┼────────
    │ FLAG                                                 │ Multiclass{2} │ Categ ⋯
    │ Avg min between sent tnx                             │ Continuous    │ Float ⋯
    │ Avg min between received tnx                         │ Continuous    │ Float ⋯
    │ Time Diff between first and last (Mins)              │ Continuous    │ Float ⋯
    │ Sent tnx                                             │ Count         │ Int64 ⋯
    │ Received Tnx                                         │ Count         │ Int64 ⋯
    │ Number of Created Contracts                          │ Count         │ Int64 ⋯
    │ Unique Received From Addresses                       │ Count         │ Int64 ⋯
    │ Unique Sent To Addresses                             │ Count         │ Int64 ⋯
    │ min value received                                   │ Continuous    │ Float ⋯
    │ max value received                                   │ Continuous    │ Float ⋯
    │ avg val received                                     │ Continuous    │ Float ⋯
    │ min val sent                                         │ Continuous    │ Float ⋯
    │ max val sent                                         │ Continuous    │ Float ⋯
    │ avg val sent                                         │ Continuous    │ Float ⋯
    │ min value sent to contract                           │ Continuous    │ Float ⋯
    │                          ⋮                           │       ⋮       │       ⋱
    └──────────────────────────────────────────────────────┴───────────────┴────────
                                                        1 column and 30 rows omitted



## Unpacking and Splitting Data

Both `MLJ` and the pure functional interface of `Imbalance` assume that the observations table `X` and target vector `y` are separate. We can accomplish that by using `unpack` from `MLJ`


```julia
y, X = unpack(df, ==(:FLAG); rng=123);
first(X, 5) |> pretty
```

Splitting the data into train and test portions is also easy using `MLJ`'s `partition` function.


```julia
(X_train, X_test), (y_train, y_test) = partition(
	(X, y),
	0.8,
	multi = true,
	shuffle = true,
	stratify = y,
	rng = Random.Xoshiro(41)
)
```

## Resampling



Before deciding to oversample, let's see how adverse is the imbalance problem, if it exists. Ideally, you may as well check if the classification model is robust to this problem.


```julia
checkbalance(y)         # comes from Imbalance
```

This signals a potential class imbalance problem. Let's consider using `SMOTE-Tomek` to resample this data. The `SMOTE-Tomek` algorithm is nothing but `SMOTE` followed by `TomekUndersampler`. We can wrap these in a pipeline along with a classification model for predictions using `BalancedModel` from `MLJBalancing`. Let's go for a `RandomForestClassifier` from `DecisionTree.jl` for the model.


```julia
import Pkg; Pkg.add("DecisionTree")
```

#### Construct the Resampling & Classification Models


```julia
oversampler = Imbalance.MLJ.SMOTE(ratios=Dict(1=>0.5), rng=Random.Xoshiro(42))
undersampler = Imbalance.MLJ.TomekUndersampler(min_ratios=Dict(0=>1.3), force_min_ratios=true)
RandomForestClassifier = @load RandomForestClassifier pkg=DecisionTree
model = RandomForestClassifier(n_trees=2, rng=Random.Xoshiro(42))
```


    RandomForestClassifier(
      max_depth = -1, 
      min_samples_leaf = 1, 
      min_samples_split = 2, 
      min_purity_increase = 0.0, 
      n_subfeatures = -1, 
      n_trees = 2, 
      sampling_fraction = 0.7, 
      feature_importance = :impurity, 
      rng = Xoshiro(0xa379de7eeeb2a4e8, 0x953dccb6b532b3af, 0xf597b8ff8cfd652a, 0xccd7337c571680d1))


#### Form the Pipeline using `BalancedModel`


```julia
balanced_model = BalancedModel(model=model, balancer1=oversampler, balancer2=undersampler)
```


    BalancedModelProbabilistic(
      model = RandomForestClassifier(
            max_depth = -1, 
            min_samples_leaf = 1, 
            min_samples_split = 2, 
            min_purity_increase = 0.0, 
            n_subfeatures = -1, 
            n_trees = 2, 
            sampling_fraction = 0.7, 
            feature_importance = :impurity, 
            rng = Xoshiro(0xa379de7eeeb2a4e8, 0x953dccb6b532b3af, 0xf597b8ff8cfd652a, 0xccd7337c571680d1)), 
      balancer1 = SMOTE(
            k = 5, 
            ratios = Dict(1 => 0.5), 
            rng = Xoshiro(0xa379de7eeeb2a4e8, 0x953dccb6b532b3af, 0xf597b8ff8cfd652a, 0xccd7337c571680d1), 
            try_preserve_type = true), 
      balancer2 = TomekUndersampler(
            min_ratios = Dict(0 => 1.3), 
            force_min_ratios = true, 
            rng = TaskLocalRNG(), 
            try_preserve_type = true))


Now we can treat `balanced_model` like any `MLJ` model.

#### Fit the `BalancedModel`


```julia
# 3. Wrap it with the data in a machine
mach_over = machine(balanced_model, X_train, y_train)

# 4. fit the machine learning model
fit!(mach_over, verbosity=0)
```


    trained Machine; does not cache data
      model: BalancedModelProbabilistic(model = RandomForestClassifier(max_depth = -1, …), …)
      args: 
        1:	Source @967 ⏎ Table{Union{AbstractVector{Continuous}, AbstractVector{Count}}}
        2:	Source @913 ⏎ AbstractVector{Multiclass{2}}



#### Validate the `BalancedModel`


```julia
cv=CV(nfolds=10)
evaluate!(mach_over, resampling=cv, measure=balanced_accuracy) 
```


    PerformanceEvaluation object with these fields:
      model, measure, operation, measurement, per_fold,
      per_observation, fitted_params_per_fold,
      report_per_fold, train_test_rows, resampling, repeats
    Extract:
    ┌─────────────────────┬──────────────┬─────────────┬─────────┬──────────────────
    │ measure             │ operation    │ measurement │ 1.96*SE │ per_fold        ⋯
    ├─────────────────────┼──────────────┼─────────────┼─────────┼──────────────────
    │ BalancedAccuracy(   │ predict_mode │ 0.93        │ 0.00757 │ [0.927, 0.936,  ⋯
    │   adjusted = false) │              │             │         │                 ⋯
    └─────────────────────┴──────────────┴─────────────┴─────────┴──────────────────
                                                                    1 column omitted



#### Compare with `RandomForestClassifier` only

To see if this represents any form of improvement, fitting and validating the original model by itself.


```julia
# 3. Wrap it with the data in a machine
mach = machine(model, X_train, y_train, scitype_check_level=0)
fit!(mach)

evaluate!(mach, resampling=cv, measure=balanced_accuracy) 
```


    PerformanceEvaluation object with these fields:
      model, measure, operation, measurement, per_fold,
      per_observation, fitted_params_per_fold,
      report_per_fold, train_test_rows, resampling, repeats
    Extract:
    ┌─────────────────────┬──────────────┬─────────────┬─────────┬──────────────────
    │ measure             │ operation    │ measurement │ 1.96*SE │ per_fold        ⋯
    ├─────────────────────┼──────────────┼─────────────┼─────────┼──────────────────
    │ BalancedAccuracy(   │ predict_mode │ 0.908       │ 0.00932 │ [0.903, 0.898,  ⋯
    │   adjusted = false) │              │             │         │                 ⋯
    └─────────────────────┴──────────────┴─────────────┴─────────┴──────────────────
                                                                    1 column omitted



Assuming normal scores, the `95%` confidence interval was `90.8±0.9` and after resampling it has become `93±0.7` which corresponds to a small improvement in accuracy.



