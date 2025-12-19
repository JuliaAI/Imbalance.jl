using Imbalance: cluster_undersample

@testset "cluster undersampler" begin
    X, y = generate_imbalanced_data(
        100,
        2;
        class_probs = [0.1, 0.6, 0.3],
        type = "Matrix",
        rng = StableRNG(192),
    )

    counts_per_class = countmap(y)
    minority_count = minimum(values(counts_per_class))

    n = Int(round(1.0 * minority_count))

    # Check implementation for nearest mode
    # Class 0 appears first in y and is processed first, so RNG state is fresh
    X_under, y_under = cluster_undersample(
        X,
        y;
        mode = "nearest",
        ratios = Dict(0 => 0.8, 1 => 1.2, 2 => 1.0),
        rng = StableRNG(122),
    )

    # Test class 0 (first processed class) against standalone kmeans
    X_c0 = X[y.==0, :]'
    n_c0 = 6  # 8 * 0.8 rounded
    centers = kmeans(Matrix(X_c0), n_c0; maxiter = 100, rng = StableRNG(122)).centers
    tree = BallTree(Matrix(X_c0))
    keep_inds, _ = knn(tree, centers, 1, true)
    keep_inds = vcat(keep_inds...)
    X_expected = X_c0[:, keep_inds]

    X_c0_under = X_under[y_under.==0, :]'
    @test X_c0_under ≈ X_expected


    # Check implementation for center mode
    X_under, y_under = cluster_undersample(
        X,
        y;
        mode = "center",
        ratios = Dict(0 => 0.8, 1 => 1.2, 2 => 1.0),
        rng = StableRNG(122),
    )

    # For center mode, the undersampled points are the cluster centers themselves
    X_c0 = X[y.==0, :]'
    expected_centers =
        kmeans(Matrix(X_c0), n_c0; maxiter = 100, rng = StableRNG(122)).centers

    @test !issubset(Set(eachrow(X_under)), Set(eachrow(X)))
    X_c0_under = X_under[y_under.==0, :]'
    @test X_c0_under ≈ expected_centers
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
