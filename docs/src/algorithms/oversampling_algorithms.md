# Oversampling Algorithms

The following table portrays the supported oversampling algorithms, whether the mechanism repeats or generates data and the supported types of data.

| Oversampling Method | Mechanism | Supported Data Types |
|:----------:|:----------:|:----------:|
| [Random Oversampler](@ref) | Repeat existing data | Continuous and/or nominal  |
| [Random Walk Oversampler](@ref) | Generate synthetic data | Continuous and/or nominal |
| [ROSE](@ref) | Generate synthetic data | Continuous |
| [SMOTE](@ref) | Generate synthetic data | Continuous |
| [Borderline SMOTE1](@ref) | Generate synthetic data | Continuous |
| [SMOTE-N](@ref) | Generate synthetic data | Nominal |
| [SMOTE-NC](@ref) | Generate synthetic data | Continuous and nominal |


## Random Oversampler

```@docs
random_oversample
```

## Random Walk Oversampler

```@docs
random_walk_oversample
```

## ROSE

```@docs
rose
```

## SMOTE

```@docs
smote
```

## Borderline SMOTE1

```@docs
borderline_smote1
```

## SMOTE-N

```@docs
smoten
```

## SMOTE-NC
    
```@docs
smotenc
```