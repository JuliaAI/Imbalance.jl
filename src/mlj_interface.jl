

macro mlj_continuous_metadata(model_name)
    quote
        MMI.metadata_pkg($model_name, name=string($model_name),
                         uuid="c709b415-507b-45b7-9a3d-1767c89fde68",
                         url="https://github.com/EssamWisam/Imbalance.jl",
                         julia=true)

        MMI.metadata_model($model_name,
                            input_scitype=Union{Table(Continuous), AbstractMatrix{Continuous}},
                            output_scitype=Union{Table(Continuous), AbstractMatrix{Continuous}},
                            path="Imbalance."*string($model_name))

        function MMI.transform_scitype(s::$model_name)
            return Tuple{Union{Table(Continuous), AbstractMatrix{Continuous}}, AbstractVector{<:Finite}}
        end
    end
end


### SMOTE
@with_kw mutable struct SMOTE <: Static
    k::Integer
    ratios::Union{Nothing, AbstractFloat, Dict{T, <:AbstractFloat}} where T = nothing
    rng::Union{Integer, AbstractRNG} = default_rng()
end;

function MMI.transform(s::SMOTE, _, X, y)
    smote(X, y; k=s.k, ratios=s.ratios, rng=s.rng)
end

@mlj_continuous_metadata SMOTE


### ROSE
@with_kw mutable struct ROSE <: Static
    s::AbstractFloat
    ratios::Union{Nothing, AbstractFloat, Dict{T, <:AbstractFloat}} where T = nothing
    rng::Union{Integer, AbstractRNG} = default_rng()
end;

function MMI.transform(r::ROSE, _, X, y)
    rose(X, y; s=r.s, ratios=r.ratios, rng=r.rng)
end

@mlj_continuous_metadata ROSE


### RandomOversampler
@with_kw mutable struct RandomOversampler <: Static
    ratios::Union{Nothing, AbstractFloat, Dict{T, <:AbstractFloat}} where T = nothing
    rng::Union{Integer, AbstractRNG} = default_rng()
end;

function MMI.transform(r::RandomOversampler, _, X, y)
    random_oversample(X, y; ratios=r.ratios, rng=r.rng)
end

@mlj_continuous_metadata ROSE