using Imbalance:
    smotenc,
    smotenc_per_class,
    generate_new_smotenc_point,
    get_random_neighbor,
    get_neighbors_mode,
    get_penalty,
    EuclideanWithPenalty,
    ERR_BAD_MIXED_COL_TYPES,
    ERR_WRNG_TREE

@testset "Testing get_penalty" begin
    X = [
        1.0 2.0 3.0 4.0
        5.0 6.0 7.0 8.0
        4.5 7.3 1.2 0.1
        9.0 10.0 11.0 12.0
    ]
    cont_inds = [1, 2, 3]

    # get the std of the first three rows
    stds = std(X[1:3, :], dims = 2)
    # get the median of the stds
    median_std = median(stds)

    @test get_penalty(X, cont_inds) ≈ median_std^2
end

# Test that generated smote point is collinear with some pair of points 
# for the continuous part and is the mode for the categorical part
@testset "generate_new_smote_point" begin
    X = [
        1.0 1.0 9.7 3.3
        4.0 3.0 9.7 5.0
        4.0 3.0 9.7 5.0
        4.0 3.0 9.7 5.5
        4.0 3.0 1.2 5.0
    ]'
    tree = BallTree(X)
    k = 3
    knn_map, _ = knn(tree, X, k + 1, true)
    cat_inds = [1, 2]
    cont_inds = [3, 4]
    new_point = vec(generate_new_smotenc_point(X, cont_inds, cat_inds, knn_map; rng))

    new_point_cont = new_point[cont_inds]
    Xcont = X[cont_inds, :]
    @test any(
        is_in_between(new_point_cont, Xcont[:, i], Xcont[:, j]) for
        i in 1:size(Xcont, 2), j in 1:size(Xcont, 2) if i != j
    )
    new_point_cat = new_point[cat_inds]
    @test new_point_cat ≈ [4.0, 3.0]
end

@testset "Testing get_neighbors_mode" begin
    # Example test data
    Xneighs = [
        1.0 2.0 3.0
        2.0 3.0 4.0
        1.0 1.0 2.0
        2.0 2.0 3.0
        3.0 3.0 3.0
        1.0 2.0 1.0
    ]'

    @test get_neighbors_mode(Xneighs, rng) ≈ [1.0, 2.0, 3.0]
end

# Test that it indeed generates n new points
@testset "smote_per_class" begin
    X = [
        1.0 1.0 9.7 3.3
        2.0 2.0 9.7 5.0
        3.0 3.0 1.2 5.0
        4.0 4.0 3.3 5.5
        5.0 5.0 1.2 1.2
    ]'
    k = 3
    n = 100
    cat_inds = [1, 2]
    cont_inds = [3, 4]
    smote_points = smotenc_per_class(X, n, cont_inds, cat_inds; k)
    @test size(smote_points, 2) == n
end

# Test bad column types error
@testset "smotenc throws error if column types are not supported" begin
    X = (
        Column1 = [1, 2, 3, 4, 5],
        Column2 = ["a", "b", "c", "d", "e"],
        Column3 = ["a", "b", "c", "d", "e"],
        Column4 = [1.0, 2.0, 3.0, 4.0, 5.0],
    )
    y = [1, 1, 1, 2, 2]
    X = Tables.columntable(X)
    # coerce first column to multiclass and last column to continuous
    # second and third column to text
    X = coerce(X, :Column1 => Multiclass, :Column4 => Continuous)
    types = ScientificTypes.schema(X).scitypes
    cat_inds = findall(x -> x <: Multiclass, types)
    cont_inds = findall(x -> x <: Union{Infinite, OrderedFactor}, types)
    @test_throws ERR_BAD_MIXED_COL_TYPES([2, 3], types[[2, 3]]) begin
        smotenc(X, y)
    end
    @test_throws ERR_WRNG_TREE("KD") begin
        X = coerce(
            X,
            :Column1 => Multiclass,
            :Column2 => Multiclass,
            :Column3 => Multiclass,
            :Column4 => Continuous,
        )
        smotenc(X, y; knn_tree = "KD")
    end
end
