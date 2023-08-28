"""
This file contains any function used by two or more of the `SMOTE` variants. 
"""

# Used by SMOTE and SMOTENC
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
Apply KNN and return the indices of the k-nearest neighbors of a given observation `x` from an observations. 

# Arguments
- `tree::KDTree`: KDTree of observations
- `x::AbstractVector`: Observation
- `k::Integer`: Number of neighbors to consider

# Returns
- `inds::AbstractVector`: Indices of the k-nearest neighbors of `x`. This is expected to include `x` itself.
"""
@memoize function memoized_knn(tree, x, k)
    # k+1 because the point is in X and will surely be a neighbor of itself
    inds, _ = knn(tree, x, k + 1, true)
    return inds
end

# Used by SMOTE, SMOTENC and SMOTEN
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
    return_all_self::Bool = false,
)
    inds = memoized_knn(tree, x, k)
    random_neighbor_index = randcols(rng, inds[2:end])[1]
    # the k+1 and the 2:end to exclude point itself
    random_neighbor = X[:, random_neighbor_index]
    # SMOTENC will further need to vote between neighbors for nominals
    return_all && return random_neighbor, X[:, inds[2:end]]
    # SMOTEN will only need to vote between neighbors for nominals
    return_all_self && return X[:, inds]
    # Base case is for SMOTE, no voting needed
    return random_neighbor
end

# Used by SMOTE, SMOTENC and SMOTEN
"""
This function is only called when n>1 and checks whether 0<k<n or not. If k<0, it throws an error.
and if k>=n, it warns the user and sets k=n-1.

# Arguments
- `k::Int`: Number of nearest neighbors to consider
- `n::Int`: Number of observations

# Returns
- `Int`: Number of nearest neighbors to consider

"""
function check_k(k, n_class)
    if k < 1
        throw(ERR_NONPOS_K(k))
    end
    if k >= n_class
        @warn WRN_K_TOO_BIG(k, n_class)
        k = n_class - 1
    end
    return k
end


# Used by SMOTENC and SMOTEN
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
    # fair voting by taking the mode over each nominal variable (row)
    # shuffle to avoid bias when breaking ties
    return [mode(shuffle(rng, row)) for row in eachrow(Xneighs)]
end



"""
Check that all columns are either categorical or continuous. If not, throw an error.

# Arguments
- `ncols::Int`: Number of columns
- `cat_inds::AbstractVector`: Indices of categorical columns
- `cont_inds::AbstractVector`: Indices of continuous columns
- `types::AbstractVector`: Types of each column

"""
function check_scitypes_smoten(ncols, cat_inds, cont_inds, types)
    bad_cols = setdiff(1:ncols, cat_inds)
    if !isempty(bad_cols)
        throw(ArgumentError(ERR_BAD_NOM_COL_TYPES(bad_cols, types[bad_cols])))
    end
    return
end

function check_scitypes_smotenc(ncols, cat_inds, cont_inds, types)
    bad_cols = setdiff(1:ncols, vcat(cat_inds, cont_inds))
    if !isempty(bad_cols)
        throw(ArgumentError(ERR_BAD_MIXED_COL_TYPES(bad_cols, types[bad_cols])))
    end
end
