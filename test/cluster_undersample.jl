using Imbalance: cluster_undersample

@testset "cluster undersampler" begin
    X, y = generate_imbalanced_data(
        100,
        2;
        probs = [0.1, 0.6, 0.3],
        type = "Matrix",
        rng = 121,
    )
    counts_per_class = countmap(y)
    minority_count = minimum(values(counts_per_class))
    X_under, y_under =
        cluster_undersample(X, y; ratios = Dict(0 => 1.0, 1 => 1.2, 2 => 1.0), rng = 121)

    # Check that the number of samples decreased correctly
    @test size(X_under, 1) == (
        Int(round(1.0 * minority_count)) +
        Int(round(1.2 * minority_count)) +
        Int(round(1.0 * minority_count))
    )
    # Check that X_under is a subset of X
    @test issubset(Set(eachrow(X_under)), Set(eachrow(X)))
    # Check implementation for nearest
    n = Int(round(1.2 * minority_count))
    X_c1 = X[y.==1, :]'
    Random.seed!(121)
    center = kmeans(X_c1, n; maxiter = 100).centers[:, 2]
    tree = BallTree(X_c1)
    i_n, _ = knn(tree, center, 1, true)
    x_n = X_c1[:, i_n[1]]
    X_c1_under = X_under[y_under.==1, :]'
    @test x_n == X_c1_under[:, 2]

    # Check implementation for center
    X_under, y_under = cluster_undersample(
        X,
        y;
        mode = "center",
        ratios = Dict(0 => 1.0, 1 => 1.2, 2 => 1.0),
        rng = 121,
    )
    @test !issubset(Set(eachrow(X_under)), Set(eachrow(X)))
    Random.seed!(121)
    @test X_under[y_under.==1, :]' == kmeans(X_c1, n; maxiter = 100).centers
end

# test that the materializer works for dataframes
@testset "materializer" begin
    X, y = generate_imbalanced_data(
        1000,
        2;
        probs = [0.2, 0.6, 0.2],
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
