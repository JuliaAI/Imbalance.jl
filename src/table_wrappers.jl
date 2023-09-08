"""
This file defines the `tablify` method which is used by all the oversampling methods
to convert functions that operate on matrices to functions that operate on tables.
"""


"""
Takes a table and returns a matrix and the column names of the table.

# Arguments
- `X`: A table to be converted to a matrix via `Tables.matrix`

# Returns
- `X`: A matrix where each row is an observation of floats
- `names`: A vector of column names
"""
function matrixify(X)
    Tables.istable(X) || throw(ERR_TABLE_TYPE(typeof(X)))
    # Get the column names using Tables.columns or the first row
    # Former is efficient for column tables, latter is efficient for row tables
    if Tables.columnaccess(X)
        columns = Tables.columns(X)
        names = Tables.columnnames(columns)
        X = Tables.matrix(X)
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
- `matrix_func`: A function that takes a matrix of observations and a vector of labels 
    and returns a matrix of oversampled observations and a vector of oversampled labels
- `X`: A table where each row is an observation of floats
- `y`: An abstract vector of class labels
- `try_perserve_type`::Bool`: Whether to convert the output back to the original table type
- `encode_func`: A function that takes the table and performs discrete encoding on it
    then returns the encoded table, a dictionary to decode it, and the indices of the categorical
    columns.
- `decode_func`: A function that takes the encoded table and the dictionary to decode it
    and returns the decoded table.

# Returns
- `Xover`: A table of the same type of X if possible (else a columntable) that 
    includes the original observations and the new observations generated by the 
    oversampling method.
- `yover`: An abstract vector of class labels that includes the original labels and 
    the new labels generated by the oversampling method.
"""
function tablify(
    matrix_func,
    X,
    y::AbstractVector;
    try_perserve_type::Bool = true,
    encode_func = X -> (X, nothing, nothing),
    decode_func = (X, d) -> (X),
    kwargs...,
)
    # 1. Encode if needed
    Xenc, decode_dict, inds = encode_func(X)

    # 2. Matrixify the table
    Xm, names = matrixify(Xenc)

    # 3. apply the algorithm logic on the matrix
    Xover, yover =
        isnothing(inds) ? matrix_func(Xm, y; kwargs...) :
        matrix_func(Xm, y, inds; kwargs...)

    # 4. Transform back to table
    Xover =  (; zip(names, eachcol(Xover))...)

    # 5. Decode if needed
    Xover = decode_func(Xover, decode_dict)

    # 6. Maintain original table type if needed
    try_perserve_type && (Xover = Tables.materializer(X)(Xover))

    return Xover, yover
end


"""
Overloads `tablify` to work with inputs where the label is one of the table columns.
"""
function tablify(
    matrix_func,
    Xy,
    y_ind::Integer;
    try_perserve_type::Bool = true,
    encode_func = X -> (X, nothing, nothing),
    decode_func = (X, d) -> (X),
    kwargs...,
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
    Xover, yover =
        isnothing(inds) ? matrix_func(Xm, y; kwargs...) :
        matrix_func(Xm, y, inds; kwargs...)

    # 5. Merge back to Xy form
    Xyover = hcat(Xover[:, 1:y_ind-1], yover, Xover[:, y_ind:end])

    # 6. Transform back to table
    Xyover =  (; zip(names, eachcol(Xyover))...)

    # 7. Decode if needed
    Xyover = decode_func(Xyover, decode_dict)

    # 8. Maintain original table type if needed
    try_perserve_type && (Xyover = Tables.materializer(Xy)(Xyover))

    return Xyover
end


"""
A function to revert oversampling by simply removing the synthetic examples. Used
with TableTransforms.
"""
function revert_oversampling(Xyover, length)
    # length is the number of original observations
    Xy = Tables.subset(Xyover, 1:length)
    return Xy
end
