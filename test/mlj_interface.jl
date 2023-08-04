using Imbalance: SMOTE, ROSE, RandomOversampler, smote, rose, random_oversample, MMI


@testset "Random Oversampler MLJ" begin
    failures, summary = MLJTestInterface.test(
        [RandomOversampler,],
        MLJTestInterface.make_multiclass()...;
        verbosity=1,
        throw=true,
        mod=@__MODULE__,
    )
    @test isempty(failures)
end

@testset "ROSE MLJ" begin
    failures, summary = MLJTestInterface.test(
        [ROSE,],
        MLJTestInterface.make_multiclass()...;
        verbosity=1,
        throw=true,
        mod=@__MODULE__,
    )
    @test isempty(failures)
end

@testset "SMOTE MLJ" begin
    failures, summary = MLJTestInterface.test(
        [SMOTE,],
        MLJTestInterface.make_multiclass()...;
        verbosity=1,
        throw=true,
        mod=@__MODULE__,
    )
    @test isempty(failures)
end

#=

# SMOTE with the MLJ interface
@testset "SMOTE with MLJ" begin
    X, y = generate_imbalanced_data(
        100,
        2;
        probs = [0.2, 0.6, 0.2],
        type = "Matrix",
        rng = 121,
    )
    mach = machine(SMOTE(k = 5))
    Xover, yover = transform(mach, X, y)
    Xover1, yover1 = smote(X, y; k = 5, rng = 121)
    @test Xover == Xover
    @test yover == yover1
end


# ROSE with the MLJ interface
@testset "ROSE with MLJ" begin
    X, y = generate_imbalanced_data(100, 2; probs = [0.2, 0.6, 0.2], type = "DF", rng = 121)
    mach = machine(ROSE(s = 0.03))
    Xover, yover = transform(mach, X, y)
    Xover1, yover1 = rose(X, y; s = 0.03, rng = 121)
    @test Xover == Xover
    @test yover == yover1
end

# Random Oversampler with the MLJ interface
@testset "Random Oversampler with MLJ" begin
    X, y = generate_imbalanced_data(100, 2; probs = [0.2, 0.6, 0.2], type = "DF", rng = 121)
    mach = machine(RandomOversampler(ratios = Dict(0 => 1.0, 1 => 1.2, 2 => 0.9)))
    Xover, yover = transform(mach, X, y)
    Xover1, yover1 =
        random_oversample(X, y; ratios = Dict(0 => 1.0, 1 => 1.2, 2 => 0.9), rng = 121)
    @test Xover == Xover
    @test yover == yover1
end

=#