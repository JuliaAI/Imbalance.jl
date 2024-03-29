using Imbalance: random_oversample

# test that random oversample indeed duplicates points in minority classes
@testset "random oversampler" begin
    X, y = generate_imbalanced_data(
        1000,
        2;
        class_probs = [0.2, 0.6, 0.2],
        type = "Matrix",
        rng = 121,
    )
    counts_per_class = countmap(y)
    majority_count = maximum(values(counts_per_class))
    Xover, yover =
        random_oversample(X, y; ratios = Dict(0 => 1.0, 1 => 1.2, 2 => 0.9), rng = 121)
    # Check that the number of samples increased correctly
    @test size(Xover, 1) == (
        Int(round(1.0 * majority_count)) +
        Int(round(1.2 * majority_count)) +
        Int(round(0.9 * majority_count))
    )
    # Check that the number of uniques is the same
    @test length(unique(Xover, dims = 1)) == length(unique(X, dims = 1))

    y = ["A", "A", "B", "A", "B"]
    X = [1 1.1 2.1;
        1 1.2 2.2;
        2 1.3 2.3;
        1 1.4 missing;
        2 1.5 2.5; ]
    Xover, yover = random_oversample(X, y) 
    @test length(unique(Xover, dims = 1)) == length(unique(X, dims = 1))

end


# test that the materializer works for dataframes
@testset "materializer" begin
    X, y =
        generate_imbalanced_data(1000, 2; class_probs = [0.2, 0.6, 0.2], type = "MatrixTable", rng = 121)
    Xover, yover =
        random_oversample(DataFrame(X), y; ratios = Dict(0 => 1.0, 1 => 1.2, 2 => 0.9), rng = 121)
    # Check that the number of samples increased correctly
    @test typeof(Xover) == DataFrame
end
