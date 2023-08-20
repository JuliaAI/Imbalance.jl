using Imbalance:
    SMOTE,
    ROSE,
    RandomOversampler,
    smote,
    rose,
    random_oversample,
    MMI,
    SMOTENC,
    smotenc,
    SMOTEN,
    smoten


@testset "Random Oversampler MLJ" begin
    failures, summary = MLJTestInterface.test(
        [RandomOversampler],
        MLJTestInterface.make_multiclass()...;
        verbosity = 1,
        throw = true,
        mod = @__MODULE__
    )
    @test isempty(failures)
end

@testset "ROSE MLJ" begin
    failures, summary = MLJTestInterface.test(
        [ROSE],
        MLJTestInterface.make_multiclass()...;
        verbosity = 1,
        throw = true,
        mod = @__MODULE__
    )
    @test isempty(failures)
end

@testset "SMOTE MLJ" begin
    failures, summary = MLJTestInterface.test(
        [SMOTE],
        MLJTestInterface.make_multiclass()...;
        verbosity = 1,
        throw = true,
        mod = @__MODULE__
    )
    @test isempty(failures)
end


@testset "SMOTENC MLJ" begin
    failures, summary = MLJTestInterface.test(
        [SMOTENC],
        MLJTestInterface.make_multiclass()...;
        verbosity = 1,
        throw = true,
        mod = @__MODULE__
    )
    @test isempty(failures)
end

# For SMOTEN, need dataset with categorical variables
