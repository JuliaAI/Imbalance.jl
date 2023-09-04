### Contains functions to be used for testing purposes
rng = StableRNG(123)

"""
Test if point a is between points b and c

# Arguments
- `a::AbstractVector`: A point
- `b::AbstractVector`: A point
- `c::AbstractVector`: A point

# Returns
- `Bool`: True if a is between b and c, false otherwise
"""
function is_in_between(a, b, c; atol = 0.01)::Bool
    dist_ab = sqrt(sum((a .- b) .^ 2))
    dist_ac = sqrt(sum((a .- c) .^ 2))
    dist_total = sqrt(sum((b .- c) .^ 2))
    return isapprox(dist_ab + dist_ac, dist_total; atol)
end
