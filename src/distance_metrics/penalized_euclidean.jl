
struct EuclideanWithPenalty <: Metric
    penalty::Float64
    cont_inds::AbstractVector{<:Int}
    cat_inds::AbstractVector{<:Int}
end



"""
This overloads the `evaluate` function of the `Metric` struct to use the modified
distance metric which adds a penalty for each pair of corresponding categorical
variables that are not equal.

# Arguments
- `d`: The modified distance metric
- `x₁`: First observation
- `x₂`: Second observation

# Returns
- `dist`: The distance between `x₁` and `x₂` using the modified distance metric
"""
function Distances.evaluate(d::EuclideanWithPenalty, x₁, x₂)
    x₁_cont, x₁_cat = x₁[d.cont_inds], x₁[d.cat_inds]
    x₂_cont, x₂_cat = x₂[d.cont_inds], x₂[d.cat_inds]
    e = sqeuclidean(x₁_cont, x₂_cont)
    h = hamming(x₁_cat, x₂_cat)
    # distance computed as described above
    dist = e + d.penalty * h
    return dist
end