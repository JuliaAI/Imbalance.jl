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
function is_in_between(a, b, c)::Bool
    dist_ab = sqrt(sum((a .- b) .^ 2))
    dist_ac = sqrt(sum((a .- c) .^ 2))
    dist_total = sqrt(sum((b .- c) .^ 2))
    return isapprox(dist_ab + dist_ac, dist_total)
end


"""
Read a Julia variable from a file
"""
function read_var(filename)
    if isfile(filename)
        open(filename, "r") do file
            return deserialize(file)
        end
    else
        error("File $filename not found. Try locally setting offline_python_test = true")
    end
end

"""
Save a Julia variable into a file
"""
function write_var(var, filename)
    open("$filename", "w") do file
        serialize(file, var)
    end
end