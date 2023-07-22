"""
    random_oversample(X, y; ratios=nothing, rng=default_rng())

Oversample a dataset given by a matrix or table of observations X and an abstract vector of 
labels y using random oversampling (repeating existing observations with replacement).

# Arguments
- `X`: A matrix or table where each row is an observation (vector) of floats
- `y`: An abstract vector of labels that correspond to the observations in X
- `ratios`: A parameter that controls the amount of oversampling to be done for each class.
    - Can be a dictionary mapping each class to the ratio of the needed number of observations for that class to the initial number of observations of the majority class.
    - Can be nothing and in this case each class will be oversampled to the size of the majority class.
    - Can be a float and in this case each class will be oversampled to the size of the majority class times the float.
- `rng::Union{AbstractRNG, Integer}`: Either an `AbstractRNG` object or an `Integer` seed to be used with `StableRNG`.

# Returns
- `Xover`: A matrix or matrix table depending on whether input is a matrix or table 
    respectively that includes original data and the new observations due to oversampling.
- `yover`: An abstract vector of labels that includes the original
    labels and the new instances of them due to oversampling.
"""
function random_oversample(
    X, y::AbstractVector; 
    ratios=nothing, rng::Union{AbstractRNG, Integer}=default_rng()
)
    # ROSE with s=0 is equivalent to random_oversample
    Xover, yover = rose(X, y; s=0.0, ratios=ratios, rng=rng)
    return Xover, yover
end