

randcols(rng::AbstractRNG, X) = X[:, rand(rng, 1:size(X, 2))]
randcols(rng::AbstractRNG, X, n) = X[:, rand(rng, 1:size(X, 2), n)]
rng_handler(rng::Integer) = StableRNG(rng)
rng_handler(rng::AbstractRNG) = rng


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

