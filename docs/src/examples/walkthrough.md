

# Introduction

In this section of the docs, we will walk you through some examples to demonstrate how you can use `Imbalance.jl` in your machine learning project. Although we focus on examples, you can learn more about how specific algorithms work by reading this series of blogposts on  [Medium](https://medium.com/towards-data-science/class-imbalance-from-random-oversampling-to-rose-517e06d7a9b).

# Prerequisites

In further examples, we will assume familiarity with the [CSV](https://csv.juliadata.org/stable/index.html), [DataFrames](https://dataframes.juliadata.org/stable/), [ScientificTypes](https://juliaai.github.io/ScientificTypes.jl/dev/) and [MLJ](https://alan-turing-institute.github.io/MLJ.jl/dev/) packages, all of which come with excellent documentation. This example is devoted to assuring and enforcing your familiarity with such packages. You can try this all examples in the docs on your browser using [Google Colab](https://githubtocolab.com/JuliaAI/Imbalance.jl/blob/dev/docs/src/examples/walkthrough.ipynb) and you can read more about that in the last section.



```julia
import Pkg;
Pkg.add(["Random", "CSV", "DataFrames", "MLJ", 
         "Imbalance", "MLJBalancing", "ScientificTypes", "HTTP"])

using Random
using CSV
using DataFrames
using MLJ
using Imbalance
using MLJBalancing
using ScientificTypes
using HTTP: download
```

## Loading Data
In this example, we will consider the [BMI dataset](https://www.kaggle.com/datasets/yasserh/bmidataset) found on Kaggle where the objective is to predict the BMI index of individuals given their gender, weight and height. 

`CSV` gives us the ability to easily read the dataset after it's downloaded as follows


```julia
download("https://raw.githubusercontent.com/JuliaAI/Imbalance.jl/dev/docs/src/examples/bmi.csv", "./")
df = CSV.read("./bmi.csv", DataFrame)

# Display the first 5 rows with DataFrames
first(df, 5) |> pretty
```

    ┌─────────┬────────┬────────┬───────┐
    │ Gender  │ Height │ Weight │ Index │
    │ String7 │ Int64  │ Int64  │ Int64 │
    │ Textual │ Count  │ Count  │ Count │
    ├─────────┼────────┼────────┼───────┤
    │ Male    │ 174    │ 96     │ 4     │
    │ Male    │ 189    │ 87     │ 2     │
    │ Female  │ 185    │ 110    │ 4     │
    │ Female  │ 195    │ 104    │ 3     │
    │ Male    │ 149    │ 61     │ 3     │
    └─────────┴────────┴────────┴───────┘


    ┌ Warning: Reading one byte at a time from HTTP.Stream is inefficient.
    │ Use: io = BufferedInputStream(http::HTTP.Stream) instead.
    │ See: https://github.com/BioJulia/BufferedStreams.jl
    └ @ HTTP.Streams /Users/essam/.julia/packages/HTTP/SN7VW/src/Streams.jl:240
    ┌ Info: Downloading
    │   source = https://raw.githubusercontent.com/JuliaAI/Imbalance.jl/dev/docs/src/examples/bmi.csv
    │   dest = ./bmi.csv
    │   progress = NaN
    │   time_taken = 0.0 s
    │   time_remaining = NaN s
    │   average_speed = 7.933 MiB/s
    │   downloaded = 8.123 KiB
    │   remaining = ∞ B
    │   total = ∞ B
    └ @ HTTP /Users/essam/.julia/packages/HTTP/SN7VW/src/download.jl:132


## Coercing Data
Typical models from `MLJ` assume that elements in each column of a table have some *scientific type* as defined by the [ScientificTypes.jl](https://juliaai.github.io/ScientificTypes.jl/dev/) package. Among the many types defined by the package, we are interested in `Multiclass`, `OrderedFactor` which fall under the `Finite` abstract type and `Continuous` and `Count` which fall under the `Infinite` abstract type.

One motivation for this package is that it's not generally obvious whether numerical data in an input table is of continuous type or categorical type given that numbers can describe both. Meanwhile, it's problematic if a model treats numerical data as say `Continuous` or `Count` when it's in reality nominal (i.e., `Multiclass`) or ordinal (i.e., `OrderedFactor`).

We can use `schema(df)` to see how each features is currently going to be interpreted by the resampling algorithms: 



```julia
ScientificTypes.schema(df)
```


    ┌────────┬──────────┬─────────┐
    │ names  │ scitypes │ types   │
    ├────────┼──────────┼─────────┤
    │ Gender │ Textual  │ String7 │
    │ Height │ Count    │ Int64   │
    │ Weight │ Count    │ Int64   │
    │ Index  │ Count    │ Int64   │
    └────────┴──────────┴─────────┘



To change encodings that are leading to incorrect interpretations (true for all variable in this example), we use the coerce method, as follows:




```julia
df = coerce(df,
            :Gender => Multiclass,
            :Height => Continuous,
            :Weight => Continuous,
            :Index => OrderedFactor)
ScientificTypes.schema(df)
```


    ┌────────┬──────────────────┬───────────────────────────────────┐
    │ names  │ scitypes         │ types                             │
    ├────────┼──────────────────┼───────────────────────────────────┤
    │ Gender │ Multiclass{2}    │ CategoricalValue{String7, UInt32} │
    │ Height │ Continuous       │ Float64                           │
    │ Weight │ Continuous       │ Float64                           │
    │ Index  │ OrderedFactor{6} │ CategoricalValue{Int64, UInt32}   │
    └────────┴──────────────────┴───────────────────────────────────┘



## Unpacking and Splitting Data

Both `MLJ` and the pure functional interface of `Imbalance` assume that the observations table `X` and target vector `y` are separate. We can accomplish that by using `unpack` from `MLJ`




```julia
y, X = unpack(df, ==(:Index); rng=123);
first(X, 5) |> pretty
```

    ┌───────────────────────────────────┬────────────┬────────────┐
    │ Gender                            │ Height     │ Weight     │
    │ CategoricalValue{String7, UInt32} │ Float64    │ Float64    │
    │ Multiclass{2}                     │ Continuous │ Continuous │
    ├───────────────────────────────────┼────────────┼────────────┤
    │ Female                            │ 173.0      │ 82.0       │
    │ Female                            │ 187.0      │ 121.0      │
    │ Male                              │ 144.0      │ 145.0      │
    │ Male                              │ 156.0      │ 74.0       │
    │ Male                              │ 167.0      │ 151.0      │
    └───────────────────────────────────┴────────────┴────────────┘



Splitting the data into train and test portions is also easy using `MLJ`'s `partition` function. `stratify=y` guarantees that the data is distributed in the same proportions as the original dataset in both splits which is more representative of the real world.



```julia
(X_train, X_test), (y_train, y_test) = partition(
	(X, y),
	0.8,
	multi = true,
	shuffle = true,
	stratify = y,
	rng = Random.Xoshiro(42)
)
```


    ((399×3 DataFrame
     Row │ Gender  Height   Weight  
         │ Cat…    Float64  Float64 
    ─────┼──────────────────────────
       1 │ Female    179.0    150.0
       2 │ Male      141.0     80.0
       3 │ Male      179.0    152.0
       4 │ Male      187.0    138.0
       5 │ Male      148.0    155.0
       6 │ Female    192.0    101.0
       7 │ Male      145.0     78.0
       8 │ Female    162.0    159.0
      ⋮  │   ⋮        ⋮        ⋮
     393 │ Female    161.0    154.0
     394 │ Female    172.0    109.0
     395 │ Female    163.0    159.0
     396 │ Female    186.0    146.0
     397 │ Male      194.0    106.0
     398 │ Female    167.0    153.0
     399 │ Female    162.0     64.0
                    384 rows omitted, 101×3 DataFrame
     Row │ Gender  Height   Weight  
         │ Cat…    Float64  Float64 
    ─────┼──────────────────────────
       1 │ Female    157.0     56.0
       2 │ Male      180.0     75.0
       3 │ Female    157.0    110.0
       4 │ Female    182.0    143.0
       5 │ Male      165.0    104.0
       6 │ Male      182.0     73.0
       7 │ Male      165.0     68.0
       8 │ Male      166.0    107.0
      ⋮  │   ⋮        ⋮        ⋮
      95 │ Male      163.0    137.0
      96 │ Female    188.0     99.0
      97 │ Female    146.0    123.0
      98 │ Male      186.0     68.0
      99 │ Female    140.0     76.0
     100 │ Female    168.0    139.0
     101 │ Male      180.0    149.0
                     86 rows omitted), (CategoricalArrays.CategoricalValue{Int64, UInt32}[5, 5, 5, 4, 5, 3, 4, 5, 5, 5  …  5, 4, 4, 5, 4, 5, 5, 3, 5, 2], CategoricalArrays.CategoricalValue{Int64, UInt32}[2, 2, 5, 5, 4, 2, 2, 4, 3, 3  …  2, 0, 0, 5, 3, 5, 2, 4, 5, 5]))


⚠️ Always split the data before oversampling. If your test data has oversampled observations then train-test contamination has occurred; novel observations will not come from the oversampling function.

## Oversampling



Before deciding to oversample, let's see how adverse is the imbalance problem, if it exists. Ideally, you may as well check if the classification model is robust to this problem.




```julia
checkbalance(y)             # comes from Imbalance
```

    0: ▇▇▇ 13 (6.6%) 
    1: ▇▇▇▇▇▇ 22 (11.1%) 
    3: ▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇ 68 (34.3%) 
    2: ▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇ 69 (34.8%) 
    4: ▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇ 130 (65.7%) 
    5: ▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇ 198 (100.0%) 


Looks like we have a class imbalance problem. Let's set the desired ratios so that the first two classes are 30%
    of the majority class, the second two are 50% of the majority class and the rest as is (ignore in the dictionary)



```julia
ratios = Dict(0=>0.3, 1=>0.3, 2=>0.5, 3=>0.5)              
```


    Dict{Int64, Float64} with 4 entries:
      0 => 0.3
      2 => 0.5
      3 => 0.5
      1 => 0.3


Let's use random oversampling to oversample the data. This particular model does not care about the scientific types of the data. It takes `X` and `y` as positional arguments and `ratios` and `rng` are the main keyword arguments



```julia
Xover, yover = random_oversample(X_train, y_train; ratios, rng=42)        
```


    (514×3 DataFrame
     Row │ Gender  Height   Weight  
         │ Cat…    Float64  Float64 
    ─────┼──────────────────────────
       1 │ Female    179.0    150.0
       2 │ Male      141.0     80.0
       3 │ Male      179.0    152.0
       4 │ Male      187.0    138.0
       5 │ Male      148.0    155.0
       6 │ Female    192.0    101.0
       7 │ Male      145.0     78.0
       8 │ Female    162.0    159.0
      ⋮  │   ⋮        ⋮        ⋮
     508 │ Female    196.0     50.0
     509 │ Male      193.0     54.0
     510 │ Male      182.0     50.0
     511 │ Male      190.0     50.0
     512 │ Male      190.0     50.0
     513 │ Male      198.0     50.0
     514 │ Male      198.0     50.0
                    499 rows omitted, CategoricalArrays.CategoricalValue{Int64, UInt32}[5, 5, 5, 4, 5, 3, 4, 5, 5, 5  …  0, 0, 0, 0, 0, 0, 0, 0, 0, 0])



```julia
checkbalance(yover)
```

    0: ▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇ 47 (29.7%) 
    1: ▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇ 47 (29.7%) 
    2: ▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇ 79 (50.0%) 
    3: ▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇ 79 (50.0%) 
    4: ▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇ 104 (65.8%) 
    5: ▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇ 158 (100.0%) 


This indeeds aligns with the desired ratios we have set earlier.

## Training the Model



Because we have scientific types setup, we can easily check what models will be able to train on our data. This should guarantee that the model we choose won't throw an error due to types after feeding it the data.



```julia
models(matching(Xover, yover))
```


    5-element Vector{NamedTuple{(:name, :package_name, :is_supervised, :abstract_type, :deep_properties, :docstring, :fit_data_scitype, :human_name, :hyperparameter_ranges, :hyperparameter_types, :hyperparameters, :implemented_methods, :inverse_transform_scitype, :is_pure_julia, :is_wrapper, :iteration_parameter, :load_path, :package_license, :package_url, :package_uuid, :predict_scitype, :prediction_type, :reporting_operations, :reports_feature_importances, :supports_class_weights, :supports_online, :supports_training_losses, :supports_weights, :transform_scitype, :input_scitype, :target_scitype, :output_scitype)}}:
     (name = CatBoostClassifier, package_name = CatBoost, ... )
     (name = ConstantClassifier, package_name = MLJModels, ... )
     (name = DecisionTreeClassifier, package_name = BetaML, ... )
     (name = DeterministicConstantClassifier, package_name = MLJModels, ... )
     (name = RandomForestClassifier, package_name = BetaML, ... )


Let's go for a decision tree form BetaML


```julia
import Pkg; Pkg.add("BetaML")
```

### Before Oversampling



```julia
# 1. Load the model
DecisionTreeClassifier = @load DecisionTreeClassifier pkg=BetaML verbosity=0

# 2. Instantiate it
model = DecisionTreeClassifier(max_depth=5, rng=Random.Xoshiro(42))

# 3. Wrap it with the data in a machine
mach = machine(model, X_train, y_train)

# 4. fit the machine learning model
fit!(mach)
```

    ┌ Info: Training machine(DecisionTreeClassifier(max_depth = 5, …), …).
    └ @ MLJBase /Users/essam/.julia/packages/MLJBase/ByFwA/src/machines.jl:492



    trained Machine; caches model-specific representations of data
      model: DecisionTreeClassifier(max_depth = 5, …)
      args: 
        1:	Source @027 ⏎ Table{Union{AbstractVector{Continuous}, AbstractVector{Multiclass{2}}}}
        2:	Source @092 ⏎ AbstractVector{OrderedFactor{6}}



### After Oversampling



```julia
# 3. Wrap it with the data in a machine
mach_over = machine(model, Xover, yover)

# 4. fit the machine learning model
fit!(mach_over)
```

    ┌ Info: Training machine(DecisionTreeClassifier(max_depth = 5, …), …).
    └ @ MLJBase /Users/essam/.julia/packages/MLJBase/ByFwA/src/machines.jl:492



    trained Machine; caches model-specific representations of data
      model: DecisionTreeClassifier(max_depth = 5, …)
      args: 
        1:	Source @592 ⏎ Table{Union{AbstractVector{Continuous}, AbstractVector{Multiclass{2}}}}
        2:	Source @711 ⏎ AbstractVector{OrderedFactor{6}}



## Evaluating the Model



To evaluate the model, we will use the balanced accuracy metric which equally account for all classes. For instance, if we have two classes and we correctly classify 100% of the examples in the first and 50% of the examples in the second then the balanced accuracy is $(100+50)/2=75%$. This holds regardless to how big or small each class is.

The `predict_mode` will return a vector of predictions given `X_test` and the fitted machine. It's different in that `predict` in not returning probablities the model assigns to each class; instead, it returns the classes with the maximum probabilities; i.e., the modes.

### Before Oversampling




```julia
y_pred = predict_mode(mach, X_test)                         

score = round(balanced_accuracy(y_pred, y_test), digits=2)
```


    0.62


### After Oversampling



```julia
y_pred_over = predict_mode(mach_over, X_test)
score = round(balanced_accuracy(y_pred_over, y_test), digits=2)
```


    0.75


## Evaluating the Model - Revisited

We have previously evaluated the model using a single point estimate of the balanced accuracy resulting in a `13%` improvement. A more precise evaluation would use cross validation to combine many different point estimates into a more precise one (their average). The standard deviation among such point estimates also allows us to quantify the uncertainty of the estimate; a smaller standard deviation would imply a smaller confidence interval at the same probability.

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
    ┌─────────────────────┬──────────────┬─────────────┬─────────┬──────────────────
    │ measure             │ operation    │ measurement │ 1.96*SE │ per_fold        ⋯
    ├─────────────────────┼──────────────┼─────────────┼─────────┼──────────────────
    │ BalancedAccuracy(   │ predict_mode │ 0.621       │ 0.0913  │ [0.593, 0.473,  ⋯
    │   adjusted = false) │              │             │         │                 ⋯
    └─────────────────────┴──────────────┴─────────────┴─────────┴──────────────────
                                                                    1 column omitted



Under the normality assumption, the `95%` confidence interval is `62.1±9.13%` which is pretty big. Let's see how it looks after oversampling.

### After Oversampling

At first glance, this seems really nontrivial since resampling will have to be performed before training the model on each fold during cross-validation. Thankfully, the `MLJBalancing` helps us avoid doing this manually by offering `BalancedModel` where we can wrap any `MLJ` classification model with an arbitrary number of `Imbalance.jl` resamplers in a pipeline that behaves like a single `MLJ` model.

In this, we must construct the resampling model via it's `MLJ` interface then pass it along with the classification model to `BalancedModel`.


```julia
# 2. Instantiate the models
oversampler = Imbalance.MLJ.RandomOversampler(ratios=ratios, rng=42)
model = DecisionTreeClassifier(max_depth=5, rng=Random.Xoshiro(42))

# 2.1 Wrap them in one model
balanced_model = BalancedModel(model=model, balancer1=oversampler)

# 3. Wrap it with the data in a machine
mach_over = machine(balanced_model, X_train, y_train, scitype_check_level=0)

# 4. fit the machine learning model
fit!(mach_over, verbosity=0)
```


    trained Machine; does not cache data
      model: BalancedModelProbabilistic(model = DecisionTreeClassifier(max_depth = 5, …), …)
      args: 
        1:	Source @099 ⏎ Table{Union{AbstractVector{Continuous}, AbstractVector{Multiclass{2}}}}
        2:	Source @071 ⏎ AbstractVector{OrderedFactor{6}}



We can easily confirm that this is equivalent to what we had earlier


```julia
predict_mode(mach_over, X_test) ==  y_pred_over
```


    true



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
    │ BalancedAccuracy(   │ predict_mode │ 0.7         │ 0.0717  │ [0.7, 0.536, 0. ⋯
    │   adjusted = false) │              │             │         │                 ⋯
    └─────────────────────┴──────────────┴─────────────┴─────────┴──────────────────
                                                                    1 column omitted



This results in an interval `70±7.2%` which can be viewed as a reasonable improvement over `62.1±9.13%`. The uncertainty in the intervals can be explained by the fact that the dataset is small with many classes.


