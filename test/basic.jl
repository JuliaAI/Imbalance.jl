using Imbalance: random_oversample

# test that random oversample indeed duplicates points in minority classes
@testset "random oversampler" begin
    X, y = generate_imbalanced_data(1000, 2; probs=[0.2, 0.6, 0.2], type="Matrix", rng=rng)
    counts_per_class = group_counts(y)
    majority_count = maximum(values(counts_per_class))
    Xover, yover = random_oversample(X, y; ratios=Dict(0=>1.0, 1=>1.2, 2=>0.9), rng=rng)
    # Check that the number of samples increased correctly
    X, Xover = X', Xover'
    @test numobs(Xover) == (Int(round(1.0 * majority_count)) + 
                            Int(round(1.2 * majority_count)) + 
                            Int(round(0.9 * majority_count)))
    # Check that the number of uniques is the same
    @test length(unique(Xover, dims=2)) == length(unique(X, dims=2))
end