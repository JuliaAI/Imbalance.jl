"""
This file contains any function used by two or more of the `SMOTE` variants. 
"""

# Used by SMOTE and SMOTENC
"""
Generate a new random observation that lies in the line joining the two observations `x₁` and `x₂`

# Arguments
- `x₁`: First observation 
- `x₂`: Second observation 
- `rng`: Random number generator

# Returns
-  New observation `x` as a vector that satisfies `x = (x₂ - x₁) * r + x₁`
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



# Used by SMOTE, SMOTENC
"""
Randomly return one of the k-nearest neighbor of a given observation `x` from an observations 
matrix `X`

# Arguments
- `X`: A matrix where each column is an observation
- `ind`: index of point for which we need random neighbor
- `knn_map`: A vector of vectors mapping each element in X by index to its nearest neighbors' indices
- `rng`: Random number generator

# Returns
- `x_randneigh`: A random observation from the k-nearest neighbors of x
"""
function get_random_neighbor(
    X::AbstractMatrix{<:Real},
    ind::Integer,
    knn_map;
    rng::AbstractRNG = default_rng(),
)
    # 1. extract the neighbors inds vector and exclude point itself
    ind_neighs = knn_map[ind][2:end]
    # 2. choose a random neighbor index
    ind_rand_neigh = ind_neighs[rand(rng, 1:length(ind_neighs))]
    # 3. return the corresponding point
    x_randneigh = X[:, ind_rand_neigh]
    return x_randneigh
end

# Used by SMOTE, SMOTENC and SMOTEN
"""
This function is only called when n>1 and checks whether 0<k<n or not. If k<0, it throws an error.
and if k>=n, it warns the user and sets k=n-1.

# Arguments
- `k`: Number of nearest neighbors to consider
- `n`: Number of observations

# Returns
-  Number of nearest neighbors to consider

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
- `Xneights`: A matrix where each row is an observation of real numbers
- `rng`: Random number generator

# Returns
-  A vector where each element is the mode of the corresponding row in `A`
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
- `ncols`: Number of columns
- `cat_inds`: Indices of categorical columns
- `cont_inds`: Indices of continuous columns
- `types`: Types of each column

"""
function check_scitypes_smoten(ncols, cat_inds, cont_inds, types)
    bad_cols = setdiff(1:ncols, cat_inds)
    if !isempty(bad_cols)
        throw(ArgumentError(ERR_BAD_NOM_COL_TYPES(bad_cols, types[bad_cols])))
    end
    return
end


"""
Check that all columns are categorical . If not, throw an error.

# Arguments
- `ncols`: Number of columns
- `cat_inds`: Indices of categorical columns
- `cont_inds`: Indices of continuous columns
- `types`: Types of each column

"""
function check_scitypes_smotenc(ncols, cat_inds, cont_inds, types)
    bad_cols = setdiff(1:ncols, vcat(cat_inds, cont_inds))
    if !isempty(bad_cols)
        throw(ArgumentError(ERR_BAD_MIXED_COL_TYPES(bad_cols, types[bad_cols])))
    end
end
