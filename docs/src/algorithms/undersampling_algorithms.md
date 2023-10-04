# Undersampling Algorithms

The following table portrays the supported undersampling algorithms, whether the mechanism deletes or generates new data and the supported types of data.

| Undersampling Method | Mechanism | Supported Data Type |
|:----------:|:----------:|:----------:|
| Random Undersampler | Delete existing data as needed | Continuous and/or nominal  |
| Cluster Undersampler | Generate new data or delete existing data | Continuous |
| ENN Undersampler | Delete existing data meeting certain conditions (cleaning) | Continuous |
| Tomek Undersampler | Delete existing data meeting certain conditions (cleaning) | Continuous |




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