"""
    random_oversample(
        X, y; 
        ratios=nothing, rng=default_rng()
    )

Oversample a dataset given by a matrix or table of observations X and an abstract vector of 
labels y using random oversampling (repeating existing observations with replacement).

# Arguments

$DOC_COMMON_INPUTS

$DOC_RATIOS_ARGUMENT

$DOC_RNG_ARGUMENT

# Returns

$DOC_COMMON_OUTPUTS
"""
function random_oversample(
    X,
    y::AbstractVector;
    ratios = nothing,
    rng::Union{AbstractRNG,Integer} = default_rng(),
)
    # ROSE with s=0 is equivalent to random_oversample
    Xover, yover = rose(X, y; s = 0.0, ratios = ratios, rng = rng)
    return Xover, yover
end


# dispatch for table inputs where y is one of the columns
function random_oversample(
    Xy,
    y_ind::Integer;
    ratios = nothing,
    rng::Union{AbstractRNG,Integer} = default_rng(),
)
    # ROSE with s=0 is equivalent to random_oversample
    Xyover = rose(Xy, y_ind; s = 0.0, ratios = ratios, rng = rng)
    return Xyover
end