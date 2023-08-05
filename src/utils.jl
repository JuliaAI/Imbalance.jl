

randcols(rng::AbstractRNG, X) = X[:, rand(rng, 1:size(X, 2))]
randcols(rng::AbstractRNG, X, n) = X[:, rand(rng, 1:size(X, 2), n)]
rng_handler(rng::Integer) = Xoshiro(rng)
rng_handler(rng::AbstractRNG) = rng


"""
Get the number of rows of a table. This implementations comes from Tables.jl as used internally there.

# Arguments
- `X`: A table

# Returns
- `Int`: Number of rows of the table
"""
function rowcount(X)
    cols = Tables.columns(X)
    names = Tables.columnnames(cols)
    isempty(names) && return 0
    return length(Tables.getcolumn(cols, names[1]))
end

"""
Return a dictionary mapping each unique value in an abstract vector to the indices of the array
where that value occurs.
"""
function group_inds(categorical_array::AbstractVector{T}) where {T}
    result = LittleDict{T,AbstractVector{Int}}()
    freeze(result)
    for (i, v) in enumerate(categorical_array)
        if !haskey(result, v)
            result[v] = []
        end
        push!(result[v], i)
    end
    return result
end
