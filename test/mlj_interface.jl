using Imbalance:
    smote,
    rose,
    random_oversample,
    random_undersample,
    cluster_undersample,
    smotenc,
    smoten,
    generate_imbalanced_data


@testset "Random Oversampler MLJ" begin
    failures, summary = MLJTestInterface.test(
        [Imbalance.MLJ.RandomOversampler],
        MLJTestInterface.make_multiclass()...;
        verbosity = 1,
        throw = true,
        mod = @__MODULE__
    )
    @test isempty(failures)
end

@testset "ROSE MLJ" begin
    failures, summary = MLJTestInterface.test(
        [Imbalance.MLJ.ROSE],
        MLJTestInterface.make_multiclass()...;
        verbosity = 1,
        throw = true,
        mod = @__MODULE__
    )
    @test isempty(failures)
end

@testset "SMOTE MLJ" begin
    failures, summary = MLJTestInterface.test(
        [Imbalance.MLJ.SMOTE],
        MLJTestInterface.make_multiclass()...;
        verbosity = 1,
        throw = true,
        mod = @__MODULE__
    )
    @test isempty(failures)
end


@testset "SMOTENC MLJ" begin
    failures, summary = MLJTestInterface.test(
        [Imbalance.MLJ.SMOTENC],
        MLJTestInterface.make_multiclass()...;
        verbosity = 1,
        throw = true,
        mod = @__MODULE__
    )
    @test isempty(failures)
end

# For SMOTEN, need dataset with categorical variables. let's (perhaps) consider a PR later.
@testset "SMOTEN MLJ" begin   
    num_rows = 100
    num_cont_feats = 0
    probs = [0.5, 0.2, 0.3]

    cat_feats_num_vals = [3, 4, 2, 5]

    X, y = generate_imbalanced_data(num_rows, num_cont_feats; probs, cat_feats_num_vals)
    X = DataFrame(X)
    X = coerce(X, autotype(X, :few_to_finite))
    smotenc_model = Imbalance.MLJ.SMOTEN(k=5, ratios=Dict(0=>1.2, 1=> 1.2, 2=>1.2), rng=42)
    mach = machine(smotenc_model)
    @test transform(mach, X, y) == smoten(X, y; k = 5, ratios = Dict(0=>1.2, 1=> 1.2, 2=>1.2), rng = 42)
end

@testset "Random Undersampler MLJ" begin
    failures, summary = MLJTestInterface.test(
        [Imbalance.MLJ.RandomUndersampler],
        MLJTestInterface.make_multiclass()...;
        verbosity = 1,
        throw = true,
        mod = @__MODULE__
    )
    @test isempty(failures)
end


@testset "Random Undersampler MLJ" begin
    failures, summary = MLJTestInterface.test(
        [Imbalance.MLJ.ClusterUndersampler],
        MLJTestInterface.make_multiclass()...;
        verbosity = 1,
        throw = true,
        mod = @__MODULE__
    )
    @test isempty(failures)
end

@testset "ENN Undersampler MLJ" begin
    failures, summary = MLJTestInterface.test(
        [Imbalance.MLJ.ENNUndersampler],
        MLJTestInterface.make_multiclass()...;
        verbosity = 1,
        throw = true,
        mod = @__MODULE__
    )
    @test isempty(failures)
end

@testset "Tomek Undersampler MLJ" begin
    failures, summary = MLJTestInterface.test(
        [Imbalance.MLJ.TomekUndersampler],
        MLJTestInterface.make_multiclass()...;
        verbosity = 1,
        throw = true,
        mod = @__MODULE__
    )
    @test isempty(failures)
end