using Imbalance: random_undersample


@testset "random undersampler" begin
    X, y = generate_imbalanced_data(
        1000,
        2;
        probs = [0.2, 0.6, 0.2],
        type = "Matrix",
        rng = 121,
    )
    counts_per_class = countmap(y)
    minority_count = minimum(values(counts_per_class))
    Xover, yover =
        random_undersample(X, y; ratios = Dict(0 => 1.0, 1 => 1.2, 2 => 1.0), rng = 121)

    # Check that the number of samples increased correctly
    @test size(Xover, 1) == (
        Int(round(1.0 * minority_count)) +
        Int(round(1.2 * minority_count)) +
        Int(round(1.0 * minority_count))
    )
    # Check that Xover is a subset of X
    @test issubset(Set(eachrow(Xover)), Set(eachrow(X)))
end


# test that the materializer works for dataframes
@testset "materializer" begin
    X, y =
        generate_imbalanced_data(1000, 2; probs = [0.2, 0.6, 0.2], type = "MatrixTable", rng = 121)
    Xover, yover =
        random_undersample(DataFrame(X), y; ratios = Dict(0 => 1.0, 1 => 1.2, 2 => 0.9), rng = 121)
    # Check that the number of samples increased correctly
    @test typeof(Xover) == DataFrame
end