"""
    random_oversample(X, y; ratios=nothing, rng=default_rng())

Oversample a dataset given by a matrix or table of observations X and a categorical vector of 
labels y using random oversampling.

# Arguments
- `X`: A matrix or table where each row is an observation (vector) of floats
- `y`: A categorical array (vector) of labels that corresponds to the observations in X
- `ratios`: A float or dictionary mapping each class to the ratio of the needed number of 
    observations of that class to the current number of observations of the majority class. 
    If nothing, then each class will be oversampled to the size of the majority class and 
    if float then each class will be oversampled to the size of the majority class times 
    the float.
- `rng::AbstractRNG`: Random number generator

# Returns
- `Xover`: A matrix or matrix table depending on whether input is a matrix or table 
    respectively that includes original data and oversampled observations.
- `yover`: A categorical array (vector) of labels that includes original
    labels and oversampled labels.
"""
function random_oversample(X,y; ratios=nothing, rng::Union{AbstractRNG, Integer}=default_rng())
    # ROSE with s=0 is equivalent to random_oversample
    Xover, yover = rose(X, y; s=0.0, ratios=ratios, rng=rng)
    return Xover, yover
end