for model_name in [:SMOTE, :ROSE, :RandomOversampler, :SMOTENC]
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
        print("Imbalance." * string($model_name))
        function MMI.transform_scitype(s::$model_name)
            return Tuple{
                Union{Table(Continuous),AbstractMatrix{Continuous}},
                AbstractVector{<:Finite},
            }
        end
    end |> eval
end

