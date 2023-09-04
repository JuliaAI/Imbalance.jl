using Imbalance:
    smote,
    rose,
    random_oversample,
    smotenc,
    smoten


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

# For SMOTEN, need dataset with categorical variables
