using Imbalance: cluster_undersample

@testset "cluster undersampler" begin
    X, y = generate_imbalanced_data(
        100,
        2;
        class_probs = [0.1, 0.6, 0.3],
        type = "Matrix",
        rng = MersenneTwister(192), 
        # changing RNG is not recommended
        # It's chosen to agree with Xoshiro (Julia 1.6 vs beyond)
        # So the dict in generic_undersample is ordered in the same way
        # in both cases. Code below assumes a specific ordering.
    )
    
    counts_per_class = countmap(y)
    minority_count = minimum(values(counts_per_class))
    
    n = Int(round(1.0 * minority_count))
    
    # Check implementation for nearest
    X_under, y_under = cluster_undersample(
        X,
        y;
        mode = "nearest",
        ratios = Dict(0 => 0.8, 1 => 1.2, 2 => 1.0),
        rng = MersenneTwister(122),
    )
    
    X_c0 = X[y.==2, :]'
    
    center = kmeans(X_c0, n; maxiter = 100, rng=MersenneTwister(122)).centers[:, 2]
    tree = BallTree(X_c0)
    i_n, _ = knn(tree, center, 1, true)
    x_n = X_c0[:, i_n[1]]
    X_c0_under = X_under[y_under .==2, :]'
    
    @test x_n ≈  X_c0_under[:, 2]
    
    
    # Check implementation for center
    X_under, y_under = cluster_undersample(
        X,
        y;
        mode = "center",
        ratios = Dict(0 => 0.8, 1 => 1.2, 2 => 1.0),
        rng = MersenneTwister(122),
    )
    
    X_c0 = X[y.==2, :]'
    
    @test !issubset(Set(eachrow(X_under)), Set(eachrow(X)))
    @test sum(X_under[y_under.==2, :]') ≈ sum(kmeans(X_c0, n; maxiter = 100, rng=MersenneTwister(122)).centers)
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
    X_under, y_under = cluster_undersample(
        DataFrame(X),
        y;
        ratios = Dict(0 => 1.0, 1 => 1.2, 2 => 0.9),
        rng = 121,
    )
    # Check that the number of samples increased correctly
    @test typeof(X_under) == DataFrame
end
