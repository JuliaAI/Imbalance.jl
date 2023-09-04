using Imbalance:
    smoten,
    smoten_per_class,
    generate_new_smoten_point,
    get_random_neighbor,
    precompute_value_encodings,
    precompute_mvdm_distances,
    ValueDifference,
    ERR_BAD_NOM_COL_TYPES




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
    knn_map, _ = knn(tree, X', k + 1)

    new_point = generate_new_smoten_point(X', knn_map; rng)

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


# Test bad column types error
@testset "smotenc throws error if column types are not supported" begin
    X = (Column1=[1, 2, 3, 4, 5],
         Column2=["a", "b", "c", "d", "e"],
         Column3=["a", "b", "c", "d", "e"],
         Column4=[1.0, 2.0, 3.0, 4.0, 5.0]
    )
    y = [1, 2, 3, 4, 5]
    X = Tables.columntable(X)
    # coerce first column to multiclass and last column to continuous
    # second and third column to text
    X = coerce(X, :Column1=>Multiclass, :Column4=>Continuous)
    types = ScientificTypes.schema(X).scitypes
    cat_inds = findall( x -> x <: Multiclass, types)
    cont_inds = findall( x -> x <: Union{Infinite, OrderedFactor}, types)    

    @test_throws ERR_BAD_NOM_COL_TYPES([2,3,4], types[[2,3,4]]) begin
        smoten(X, y)
    end
end