"""
This function is a generic implementation of oversampling methods that apply oversampling
logic independently to each class. 

# Arguments
- `X::AbstractMatrix`: A matrix or table where each row is an observation of floats
- `y::AbstractVector`: An abstract vector of class labels
- `oversample_per_class`: A function that takes a matrix of observations, a number
    of new observations to generate, and possibly other keyword arguments and 
    returns a matrix where each row is a new observation generated by the oversampling method
$DOC_RATIOS_ARGUMENT
- `kwargs`: Keyword arguments to pass to `oversample_per_class`

# Returns
- `Xover`: A matrix that includes the original observations and the 
    new observations generated by the oversampling method for each class.
- `yover`: An abstract vector of class labels that includes the original labels and the 
    new labels generated by the oversampling method for each class.
"""
function generic_oversample(
    X::AbstractMatrix{<:Real},
    y::AbstractVector,
    oversample_per_class,
    args...;
    ratios = nothing,
    kwargs...,
)
    X = transpose(X)
    # Get maps from labels to indices and the needed counts
    label_inds = group_inds(y)
    extra_counts = get_class_counts(y, ratios)
    # Apply oversample per class on each set of points belonging to the same class
    for (label, inds) in label_inds
        X_label = @view X[:, inds]
        n = extra_counts[label]
        n == 0 && continue
        Xnew = oversample_per_class(X_label, n, args...; kwargs...)
        ynew = fill(label, size(Xnew, 2))
        X = hcat(X, Xnew)
        y = vcat(y, ynew)
    end
    Xover = transpose(X)
    yover = y
    return Xover, yover
end

"""
Takes a table and returns a matrix and the column names of the table.

# Arguments
- `X::AbstractMatrix`: A matrix or table where each row is an observation of floats

# Returns
- `X::AbstractMatrix`: A matrix where each row is an observation of floats
- `names::AbstractVector`: A vector of column names
"""
const ERR_TABLE_TYPE(t) = "Error: expected a table or matrix but got a data of type $t."
function matrixify(X)
    Tables.istable(X) || throw(ERR_TABLE_TYPE(typeof(X)))
    if Tables.columnaccess(X)
        columns = Tables.columns(X)
        names = Tables.columnnames(columns)
        X = Tables.matrix(columns)
    else
        iter = iterate(Tables.rows(X))
        names = iter === nothing ? () : Tables.columnnames(first(iter))
        X = Tables.matrix(X)
    end
    return X, names
end

"""
Takes a function that takes X, y and keyword arguments but only works with abstract matrices 
and generalizes it to work with tables. To do this, it converts the table to a matrix, 
applies the function, then converts the matrix back to a table.

# Arguments
- `matrix_func::Function`: A function that takes a matrix of observations and a vector of labels 
    and returns a matrix of oversampled observations and a vector of oversampled labels
- `X::AbstractMatrix`: A table where each row is an observation of floats
- `y::AbstractVector`: An abstract vector of class labels
- `materialize::Bool`: Whether to convert the output back to the original table type
- `encode_func::Function`: A function that takes the table and performs discrete encoding on it
    then returns the encoded table, a dictionary to decode it, and the indices of the categorical
    columns.
- `decode_func::Function`: A function that takes the encoded table and the dictionary to decode it
    and returns the decoded table.

# Returns
- `Xover`: A table of the same type of X if possible (else a columntable) that 
    includes the original observations and the new observations generated by the 
    oversampling method.
- `yover`: An abstract vector of class labels that includes the original labels and 
    the new labels generated by the oversampling method.
"""
function tablify(
    matrix_func::Function,
    X,
    y::AbstractVector;
    materialize::Bool = true,
    encode_func::Function = X -> (X, nothing, nothing),
    decode_func::Function = (X, d) -> (X),
    kwargs...,
)
    # 1. Encode if needed
    Xenc, decode_dict, inds = encode_func(X)

    # 2. Matrixify the table
    Xm, names = matrixify(Xenc)

    # 3. apply the algorithm logic on the matrix
    Xover, yover = isnothing(inds) ? matrix_func(Xm, y; kwargs...) : matrix_func(Xm, y, inds; kwargs...)

    # 4. Transform back to table
    Xover = Tables.table(Xover; header = names)

    # 5. Decode if needed
    Xover = decode_func(Xover, decode_dict)

    # 6. Maintain original table type if needed
    materialize && (Xover = Tables.materializer(X)(Xover))

    return Xover, yover
end


"""
Overloads `tablify` to work with inputs where the label is one of the table columns.
"""
function tablify(
    matrix_func::Function, 
    Xy, 
    y_ind::Int; 
    materialize::Bool = true, 
    encode_func::Function = X -> (X, nothing, nothing),
    decode_func::Function = (X, d) -> (X),
    kwargs...
)
    # 1. Encode if needed
    Xyenc, decode_dict, inds = encode_func(Xy)

    # 2. Matrixify the table
    Xym, names = matrixify(Xyenc)

    # 3. Before proceeding as usual, split the matrix into X and y
    Xm = @view Xym[:, 1:end.!=y_ind]
    y = @view Xym[:, y_ind]

    # 3.1 Fix inds after removing y_ind
    if !isnothing(inds)
        inds = filter(ind -> ind != y_ind, inds)    # y_ind is no longer a categorical variable
        inds = inds .- (inds .> y_ind)              # Increment the indices of the columns after y_ind
    end

    # 4. Apply the algorithm logic on the matrix
    Xover, yover = isnothing(inds) ? matrix_func(Xm, y; kwargs...) : matrix_func(Xm, y, inds; kwargs...)

    # 5. Merge back to Xy form
    Xyover = hcat(Xover[:, 1:y_ind-1], yover, Xover[:, y_ind:end])

    # 6. Transform back to table
    Xyover = Tables.table(Xyover; header = names)

    # 7. Decode if needed
    Xyover = decode_func(Xyover, decode_dict)

    # 8. Maintain original table type if needed
    materialize && (Xyover = Tables.materializer(Xy)(Xyover))

    return Xyover
end
# materialize => try_perserve_type (will think again)

"""
A function to revert oversampling by simply removing the synthetic examples. Used
with TableTransforms.
"""
function revert_oversampling(Xyover, length)
    Xy = Tables.subset(Xyover, 1:length)
    return Xy
end
