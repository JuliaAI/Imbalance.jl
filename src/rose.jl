"""
Assuming that all the observations in the observation matrix X belong to the same class, 
generate n new observations for that class using ROSE.

# Arguments
- `X::AbstractMatrix`: A matrix where each row is an observation of floats
- `n::Int`: Number of new observations to generate
- `s::float`: A parameter that proportionally controls the bandwidth of the Gaussian kernel
- `rng::AbstractRNG`: Random number generator

# Returns
- `AbstractMatrix`: A matrix where each row is a new observation generated by ROSE
"""
function rose_per_class(
    X::AbstractMatrix{<:AbstractFloat}, n::Int; 
    s::AbstractFloat=1.0,  rng::AbstractRNG=default_rng()
)
    # sample n rows from X
    Xnew = randrows(rng, X, n)
    # For s == 0 this is just random oversampling
    if s == 0.0 return Xnew end
    # compute the standard deviation column-wise
    σs = vec(std(Xnew, dims=2))
    d = size(Xnew, 2)
    N = size(Xnew, 1)
    h = (4/((d+2)*N))^(1/(d+4))
    # make a diagonal matrix of the result
    H = Diagonal(σs * s * h)
    # generate standard normal samples of same dimension of Xnew
    XSnew = randn(rng, size(Xnew))
    # matrix multiply the diagonal matrix by XSnew
    XSnew =  H * XSnew 
    # add Xnew and XSnew
    Xnew += XSnew
    # return the result
    return Xnew
end

"""
    rose(X::AbstractMatrix{<:AbstractFloat}, y; s::AbstractFloat=0.1, ratios=nothing, rng::AbstractRNG=default_rng())
    rose(X, y; s::AbstractFloat=0.1, ratios=nothing, rng::AbstractRNG=default_rng())

Oversample a dataset given by a matrix or table of observations X and an abstract
vector of labels y using ROSE.

# Arguments

- `X`: A matrix or table where each row is an observation (vector) of floats
- `y`: An abstract vector of labels that correspond to the observations in X
- `s::float`: A parameter that proportionally controls the bandwidth of the Gaussian kernel
- `ratios`: A parameter that controls the amount of oversampling to be done for each class.
    - Can be a dictionary mapping each class to the ratio of the needed number of observations for that class to the initial number of observations of the majority class.
    - Can be nothing and in this case each class will be oversampled to the size of the majority class.
    - Can be a float and in this case each class will be oversampled to the size of the majority class times the float.
- `rng::Union{AbstractRNG, Integer}`: Either an `AbstractRNG` object or an `Integer` seed to be used with `StableRNG`.

# Returns
- `Xover`: A matrix or table like X (if possible, else a columntable) depending on whether X is a matrix or table 
    respectively that includes original data and the new observations due to oversampling.
- `yover`: An abstract vector of labels that includes the original
    labels and the new instances of them due to oversampling.
"""
function rose(
    X::AbstractMatrix{<:AbstractFloat}, y::AbstractVector; 
    s::AbstractFloat=0.1, ratios=nothing, rng::Union{AbstractRNG, Integer}=default_rng()
)
    rng = rng_handler(rng)
    Xover, yover = generic_oversample(X, y, rose_per_class; s, ratios, rng)
    return Xover, yover
end

function rose(
    X, y::AbstractVector; 
    s::AbstractFloat=0.1, ratios=nothing, rng::Union{AbstractRNG, Integer}=default_rng()
)
    Xover, yover = tablify(rose, X, y; s, ratios, rng)
    return Xover, yover
end