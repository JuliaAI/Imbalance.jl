# Introduction

In this section of the docs, we will walk you through some examples to demonstrate how you can use `Imbalance.jl` in your machine learning project. Although we focus on examples, you can learn more about how specific algorithms work by reading this series of blogposts on  [Medium](https://medium.com/towards-data-science/class-imbalance-from-random-oversampling-to-rose-517e06d7a9b).

# Prerequisites

In further examples, we will assume familiarity with the [CSV](https://csv.juliadata.org/stable/index.html), [DataFrames](https://dataframes.juliadata.org/stable/), [ScientificTypes](https://juliaai.github.io/ScientificTypes.jl/dev/) and [MLJ](https://alan-turing-institute.github.io/MLJ.jl/dev/) packages, all of which come with excellent documentation. This example is devoted to assuring and enforcing your familiarity with such packages. You can try this all examples in the docs on your browser using [Google Colab](https://colab.research.google.com/github/JuliaAI/Imbalance.jl/blob/main/examples/walkthrough.ipynb) and you can read more about that in the last section.


```julia
using Random
using CSV
using DataFrames
using MLJ
using Imbalance
using ScientificTypes
```

## Loading Data
In this example, we will consider the [BMI dataset](https://www.kaggle.com/datasets/yasserh/bmidataset) found on Kaggle where the objective is to predict the BMI index of individuals given their gender, weight and height. 

`CSV` gives us the ability to easily read the dataset after it's downloaded as follows


```julia
df = CSV.read("datasets/bmi.csv", DataFrame)

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
train_inds, test_inds = partition(
    eachindex(y), 0.8, shuffle=true, stratify=y, rng=Random.Xoshiro(42))
X_train, X_test = X[train_inds, :], X[test_inds, :]
y_train, y_test = y[train_inds], y[test_inds]

```


    (CategoricalArrays.CategoricalValue{Int64, UInt32}[5, 5, 5, 4, 5, 3, 4, 5, 5, 5  …  5, 4, 4, 5, 4, 5, 5, 3, 5, 2], CategoricalArrays.CategoricalValue{Int64, UInt32}[2, 2, 5, 5, 4, 2, 2, 4, 3, 3  …  2, 0, 0, 5, 3, 5, 2, 4, 5, 5])


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
Xover, yover = random_oversample(X, y; ratios, rng=42)        
```

    Progress:  33%|█████████████▋                           |  ETA: 0:00:01[K
    [A


    (644×3 DataFrame
     Row │ Gender  Height   Weight  
         │ Cat…    Float64  Float64 
    ─────┼──────────────────────────
       1 │ Female    173.0     82.0
       2 │ Female    187.0    121.0
       3 │ Male      144.0    145.0
       4 │ Male      156.0     74.0
       5 │ Male      167.0    151.0
       6 │ Female    146.0    147.0
       7 │ Female    157.0    153.0
       8 │ Male      187.0    140.0
      ⋮  │   ⋮        ⋮        ⋮
     638 │ Female    183.0     50.0
     639 │ Female    163.0     57.0
     640 │ Female    190.0     50.0
     641 │ Male      181.0     51.0
     642 │ Male      188.0     54.0
     643 │ Female    191.0     54.0
     644 │ Male      198.0     50.0
                    629 rows omitted, CategoricalArrays.CategoricalValue{Int64, UInt32}[2, 4, 5, 4, 5, 5, 5, 5, 5, 2  …  0, 0, 0, 0, 0, 0, 0, 0, 0, 0])



```julia
checkbalance(yover)
```

    0: ▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇ 59 (29.8%) 
    1: ▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇ 59 (29.8%) 
    2: ▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇ 99 (50.0%) 
    3: ▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇ 99 (50.0%) 
    4: ▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇ 130 (65.7%) 
    5: ▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇ 198 (100.0%) 


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

        Updating registry at `~/.julia/registries/General.toml`
       Resolving package versions...
      No Changes to `~/Documents/GitHub/Imbalance.jl/Project.toml`
      No Changes to `~/Documents/GitHub/Imbalance.jl/Manifest.toml`


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
    └ @ MLJBase /Users/essam/.julia/packages/MLJBase/0rn2V/src/machines.jl:492



    trained Machine; caches model-specific representations of data
      model: DecisionTreeClassifier(max_depth = 5, …)
      args: 
        1:	Source @636 ⏎ Table{Union{AbstractVector{Continuous}, AbstractVector{Multiclass{2}}}}
        2:	Source @264 ⏎ AbstractVector{OrderedFactor{6}}



### After Oversampling


```julia
# 3. Wrap it with the data in a machine
mach_over = machine(model, Xover, yover)

# 4. fit the machine learning model
fit!(mach_over)
```

    ┌ Info: Training machine(DecisionTreeClassifier(max_depth = 5, …), …).
    └ @ MLJBase /Users/essam/.julia/packages/MLJBase/0rn2V/src/machines.jl:492



    trained Machine; caches model-specific representations of data
      model: DecisionTreeClassifier(max_depth = 5, …)
      args: 
        1:	Source @373 ⏎ Table{Union{AbstractVector{Continuous}, AbstractVector{Multiclass{2}}}}
        2:	Source @042 ⏎ AbstractVector{OrderedFactor{6}}



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


    0.77


# Google Colab

It is possible to run this tutorial and others in the examples section on Google Colab.
- Click the Colab icon link as your hover on the example
- Paste and run the following in the first cell

```julia
%%capture
%%shell
if ! command -v julia 3>&1 > /dev/null
then
    wget -q 'https://julialang-s3.julialang.org/bin/linux/x64/1.7/julia-1.7.2-linux-x86_64.tar.gz' \
        -O /tmp/julia.tar.gz
    tar -x -f /tmp/julia.tar.gz -C /usr/local --strip-components 1
    rm /tmp/julia.tar.gz
fi
julia -e 'using Pkg; pkg"add IJulia; precompile;"'
echo 'Done'
```

- Change the runtime to Julia from the toolbar
- `Pkg.add` Imbalance and any needed packages (those being used)
- Click the folder icon on the left, make a `datasets` folder and drag and drop it in there
- Run the notebook

Sincere thanks to [Julia-on-Colab](https://github.com/Dsantra92/Julia-on-Colab) for making this possible



