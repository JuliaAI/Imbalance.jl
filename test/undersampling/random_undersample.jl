using Imbalance: random_undersample

@testset "random undersampler" begin
    X, y = generate_imbalanced_data(
        1000,
        2;
        class_probs = [0.2, 0.6, 0.2],
        type = "Matrix",
        rng = 121,
    )
    counts_per_class = countmap(y)
    minority_count = minimum(values(counts_per_class))
    X_under, y_under =
        random_undersample(X, y; ratios = Dict(0 => 1.0, 1 => 1.2, 2 => 1.0), rng = 121)

    # Check that the number of samples increased correctly
    @test size(X_under, 1) == (
        Int(round(1.0 * minority_count)) +
        Int(round(1.2 * minority_count)) +
        Int(round(1.0 * minority_count))
    )
    # Check that X_under is a subset of X
    @test issubset(Set(eachrow(X_under)), Set(eachrow(X)))

    y = ["A", "A", "B", "A", "B"]
    X = [1 1.1 2.1;
        1 1.2 2.2;
        2 1.3 2.3;
        1 1.4 missing;
        2 1.5 2.5; ]
    X_under, y_under = random_undersample(X, y) 
    @test issubset(Set(eachrow(X_under)), Set(eachrow(X)))
end

# test that the materializer works for dataframes
@testset "materializer" begin
    X, y = generate_imbalanced_data(
        1000,
        2;
        class_probs = [0.2, 0.6, 0.2],
        type = "MatrixTable",
        rng = 121,
    )
    X_under, y_under = random_undersample(
        DataFrame(X),
        y;
        ratios = Dict(0 => 1.0, 1 => 1.2, 2 => 0.9),
        rng = 121,
    )
    # Check that the number of samples increased correctly
    @test typeof(X_under) == DataFrame
end
