for model_name in [:SMOTE, :ROSE, :RandomOversampler]
    quote
        MMI.metadata_pkg(
            $model_name,
            name = "Imbalance",
            package_uuid = "c709b415-507b-45b7-9a3d-1767c89fde68",
            package_url = "https://github.com/EssamWisam/Imbalance.jl",
            is_pure_julia = true,
        )

        MMI.metadata_model(
            $model_name,
            input_scitype = Union{Table(Continuous),AbstractMatrix{Continuous}},
            output_scitype = Union{Table(Continuous),AbstractMatrix{Continuous}},
            load_path = "Imbalance." * string($model_name),
        )
        function MMI.transform_scitype(s::$model_name)
            return Tuple{
                Union{Table(Continuous),AbstractMatrix{Continuous}},
                AbstractVector{<:Finite},
            }
        end
    end |> eval
end



### SMOTEN
MMI.metadata_pkg(
    SMOTEN,
    name = "Imbalance",
    package_uuid = "c709b415-507b-45b7-9a3d-1767c89fde68",
    package_url = "https://github.com/EssamWisam/Imbalance.jl",
    is_pure_julia = true,
)

MMI.metadata_model(
    SMOTEN,
    input_scitype = Union{Table(Multiclass),AbstractMatrix{Multiclass}},
    output_scitype = Union{Table(Multiclass),AbstractMatrix{Multiclass}},
    load_path = "Imbalance.SMOTEN",
)

function MMI.transform_scitype(s::SMOTEN)
    return Tuple{
        Union{Table(Multiclass),AbstractMatrix{Multiclass}},
        AbstractVector{<:Finite},
    }
end


### SMOTENC
MMI.metadata_pkg(
    SMOTENC,
    name = "Imbalance",
    package_uuid = "c709b415-507b-45b7-9a3d-1767c89fde68",
    package_url = "https://github.com/EssamWisam/Imbalance.jl",
    is_pure_julia = true,
)

MMI.metadata_model(
    SMOTENC,
    input_scitype = Union{Table(Union{Infinite, OrderedFactor, Multiclass}),
    AbstractMatrix{Union{Infinite, OrderedFactor, Multiclass}}},
    output_scitype = Union{Table(Union{Infinite, OrderedFactor, Multiclass}),
    AbstractMatrix{Union{Infinite, OrderedFactor, Multiclass}}},
    load_path = "Imbalance.SMOTEN",
)

function MMI.transform_scitype(s::SMOTENC)
    return Tuple{
        Union{Table(Union{Infinite, OrderedFactor, Multiclass}),AbstractMatrix{Union{Infinite, OrderedFactor, Multiclass}}},
        AbstractVector{<:Finite},
    }
end