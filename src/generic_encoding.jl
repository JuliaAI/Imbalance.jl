"""
This files defines the `generic_encoder` and `generic_decoder` functions which is used by all the 
oversampling methods that accept categorical features such as `SMOTE-N`,`SMOTE-NC` and `Random Oversample`.
"""

"""
Apply label encoding to the categorical columns of a table. A categorical column is defined
as any column with the scitype `Multiclass` or `OrderedFactor`.

# Arguments
- `X`: A table where each row is an observation which has some categorical columns
- error_checker: A function that checks if the input table has correct scitypes
- `return_cat_inds::Bool`: If true, the function also returns a vector with the categorical indices

# Returns
- `Xenc`: A column table where the categorical columns have been replaced by their label encoded
    versions
- `decode_dict`: A dictionary used in label decoding of the table
- `cat_inds`: A vector with the indices of the categorical columns

"""
function generic_encoder(X; error_checker=(args...)->nothing, return_cat_inds = false)
    # 1. Find the categorical and continuous columns
    types = ScientificTypes.schema(X).scitypes
    cat_inds = findall(x -> x <: Finite, types)
    cont_inds = findall(x -> x <: Infinite, types)
    error_checker(length(types), cat_inds, cont_inds, types)

    # 2. Setup the encode and decode transforms for categotical columns
    encode_dict = Dict{Int,Function}()
    decode_dict = Dict{Int,Function}()

    columns = Tables.columns(X)
    for c in cat_inds
        column = collect(Tables.getcolumn(columns, c))
        decode_dict[c] = x -> CategoricalDistributions.decoder(column)(round(Int, x))
        encode_dict[c] = x -> CategoricalDistributions.int(x)
    end

    # 3. Encode the data
    Xenc = X |> TableOperations.transform(encode_dict) |> Tables.columntable
    # TODO: Remove Tables.columntable once https://github.com/JuliaData/TableOperations.jl/issues/32 is resolved

    # 4. SMOTE-N encoder need not pass cat_inds to tablify
    !return_cat_inds && return Xenc, decode_dict, nothing
    return Xenc, decode_dict, cat_inds
end


"""
Decode the label encoded categorical columns of a table.

# Arguments
- `Xover`: A table where each row is an observation which has some label-encoded categorical columns
- `decode_dict`: A dictionary of functions to decode the label-encoded categorical columns

# Returns
- `Xover`: A column table where the categorical columns is decoded back to their original values
"""
function generic_decoder(Xover, decode_dict)
    Xover = Xover |> TableOperations.transform(decode_dict)
    return Xover
end