# Undersampling Algorithms

The following table portrays the supported undersampling algorithms, whether the mechanism deletes or generates new data and the supported types of data.

| Undersampling Method | Mechanism | Supported Data Types |
|:----------:|:----------:|:----------:|
| [Random Undersampler](@ref) | Delete existing data as needed | Continuous and/or nominal  |
| [Cluster Undersampler](@ref) | Generate new data or delete existing data | Continuous |
| [Edited Nearest Neighbors Undersampler](@ref) | Delete existing data meeting certain conditions (cleaning) | Continuous |
| [Tomek Links Undersampler](@ref) | Delete existing data meeting certain conditions (cleaning) | Continuous |




## Random Undersampler
    
```@docs
random_undersample
```

## Cluster Undersampler
    
```@docs
cluster_undersample
```

## Edited Nearest Neighbors Undersampler
    
```@docs
enn_undersample
```

## Tomek Links Undersampler
    
```@docs
tomek_undersample
```