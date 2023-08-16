"""
This file contains any function used by two or more of the SMOTE variants. 
"""


"""
Generate a new random observation that lies in the line joining the two observations `x₁` and `x₂`

# Arguments
- `x₁::AbstractVector`: First observation 
- `x₂::AbstractVector`: Second observation 
- `rng::AbstractRNG`: Random number generator

# Returns
- `AbstractMatrix`: New observation `x` as a row vector that satisfies `x = (x₂ - x₁) * r + x₁`
    where `r`` is a random number between `0` and `1`
"""
function get_collinear_point(
    x₁::AbstractVector,
    x₂::AbstractVector;
    rng::AbstractRNG = default_rng(),
)
    r = rand(rng)
    # Equivalent to (x₂  .- x₁ ) .* r .+ x₁  but avoids allocating a new vector
    return @. (1 - r) * x₁ + r * x₂
end


"""
Randomly return one of the k-nearest neighbor of a given observation `x` from an observations 
matrix `X` represented by a k-d tree

# Arguments
- `X::AbstractMatrix`: A matrix where each row is an observation
- `tree`: A k-d tree representation of the observations matrix `X`
- `x::AbstractVector`: An observation
- `k::Int`: Number of nearest neighbors to consider
- `rng::AbstractRNG`: Random number generator
- `return_all::Bool`: If true, return the random neighbor and all the k-nearest neighbors of `x`
- `return_all_self::Bool`: If true (and return_all is false), return all the k-nearest neighbors of `x` 
    including `x` itself

# Returns
- `AbstractVector`: A random observation from the k-nearest neighbors of x
"""
function get_random_neighbor(
    X::AbstractMatrix{<:Real},
    tree,
    x;
    k::Int = 5,
    rng::AbstractRNG = default_rng(),
    return_all::Bool = false,
    return_all_self::Bool = false
)
    inds, _ = knn(tree, x, k + 1, true)
    # Need to deal with that the first neighbor is the point itself; hence, the k+1 and the 2:end
    random_neighbor_index = randcols(rng, inds[2:end])[1]
    random_neighbor = X[:, random_neighbor_index]
    return_all && return random_neighbor, X[:, inds[2:end]]
    return_all_self && return X[:, inds]
    return random_neighbor
end




"""
Find the mode of each row in a matrix and return the result as a vector. If multiple ⊧
exist, choose one of them randomly.

# Arguments
- `A::AbstractMatrix`: A matrix where each row is an observation of real numbers
- `rng::AbstractRNG`: Random number generator

# Returns
- `AbstractVector`: A vector where each element is the mode of the corresponding row in `A`
"""
function get_neighbors_mode(
    Xneighs::AbstractMatrix{<:Real},
    rng::AbstractRNG = default_rng(),
)
    return [mode(shuffle(rng, row)) for row in eachrow(Xneighs)]
end
