using Imbalance:
    smoten,
    smoten_per_class,
    generate_new_smoten_point,
    get_random_neighbor,
    precompute_value_encodings,
    precompute_mvdm_distances,
    ValueDifference


# TODO: make test work with Github actions
@testset "MVDM Encoding and Distance" begin
    Conda.add("imbalanced-learn")
    pyimport_conda("imblearn", "imbalanced-learn")
    # Import numpy and other libraries
    np = pyimport("numpy")
    imblearn = pyimport("imblearn")
    fromsklearn = pyimport("sklearn.preprocessing")

    # Set the dimensions of the matrix
    rows = 50
    cols = 8

    # Generate a random matrix X and vector y in Python
    X = np.random.randint(1, 11, size=(rows, cols))
    y = np.random.randint(1, 6, size=rows)

    mvdm_encoder_python = imblearn.metrics.pairwise.ValueDifferenceMetric().fit(X, y)

    mvdm_encoder, num_cats_per_column = precompute_value_encodings(X, y)
    for i ∈ 1:8
        @test mvdm_encoder_python.proba_per_class_[i][2:end,:] ≈ mvdm_encoder[i]'
    end

    all_pairwise_mvdm = precompute_mvdm_distances(mvdm_encoder, num_cats_per_column)

    V = ValueDifference(all_pairwise_mvdm)
    D = pairwise(V, X, X, dims=1)
    @test sum(pairwise(V, X, X, dims=1)) ≈ sum(D)
end


# Test that generated smote point is collinear with some pair of points
@testset "generate_new_smote_point" begin
    X = [
        1 1 1 1
        2 3 1 2
        3 3 1 2
        4 4 1 2
        4 2 2 2
    ]

    y = [2, 1, 1, 1, 1]
    k = 4
    mvdm_encoder, num_categories_per_col = precompute_value_encodings(X, y)
    all_pairwise_vdm = precompute_mvdm_distances(mvdm_encoder, num_categories_per_col)
    metric = ValueDifference(all_pairwise_vdm)
    tree = BruteTree(X', metric)
    knn_matrix, _ = knn(tree, X', k + 1)

    new_point = generate_new_smoten_point(X', knn_matrix; k, rng)

    @test new_point == [4, 3, 1, 2]
end

# Test that it indeed generates n new points
@testset "smoten_per_class" begin
    X = [
        1 1 1 1
        2 3 1 2
        3 3 1 2
        4 4 1 2
        4 2 2 2
    ]
    y = [2, 1, 1, 1, 1]
    k = 3
    n = 100
    mvdm_encoder, num_categories_per_col = precompute_value_encodings(X, y)
    all_pairwise_vdm = precompute_mvdm_distances(mvdm_encoder, num_categories_per_col)
    smote_points = smoten_per_class(X', n, all_pairwise_vdm; k, rng)
    @test size(smote_points, 2) == n
end
