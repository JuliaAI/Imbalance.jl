using Imbalance:
    smoten,
    smoten_per_class,
    generate_new_smoten_point,
    get_random_neighbor,
    precompute_pairwise_value_difference,
    ValueDifference

@testset "Precompute Pairwise VDM" begin

    X = 
        [ 1  2  1
        2  3  4
        3  2  1
        1  1  4
        1  4  5
        2  3  3
        3  2  2 ]

    y = [1 
        2 
        3 
        3 
        4 
        4 
        1]


    res = 
    [
    [0.3333333333333333 0.0 0.5
    0.0 0.5 0.0
    0.3333333333333333 0.0 0.5
    0.3333333333333333 0.5 0.0],

    [0.0 0.6666666666666666 0.0 0.0
    0.0 0.0 0.5 0.0
    1.0 0.3333333333333333 0.0 0.0
    0.0 0.0 0.5 1.0],
    
    [0.5  1.0  0.0  0.0  0.0
    0.0  0.0  0.0  0.5  0.0
    0.5  0.0  0.0  0.5  0.0
    0.0  0.0  1.0  0.0  1.0]
    ]

    all_pairwise_vdm = precompute_pairwise_value_difference(X, y)

    dist = Cityblock()
    @test all_pairwise_vdm[1] == pairwise(dist, res[1], res[1], dims=2)
    @test all_pairwise_vdm[2] == pairwise(dist, res[2], res[2], dims=2)
    @test all_pairwise_vdm[3] == pairwise(dist, res[3], res[3], dims=2)

    ValueDiff = ValueDifference(all_pairwise_vdm)

    @test Distances.evaluate(ValueDiff, [1, 2, 3], [3, 4, 5]) == all_pairwise_vdm[1][1, 3]^2 + all_pairwise_vdm[2][2, 4]^2 + all_pairwise_vdm[3][3, 5]^2
    @test Distances.evaluate(ValueDiff, [1, 2, 3], [3, 4, 5]) == all_pairwise_vdm[1][3, 1]^2 + all_pairwise_vdm[2][4, 2]^2 + all_pairwise_vdm[3][5, 3]^2

end


# Test that a random neighbor is indeed one of the nearest neighbors
@testset "get_random_neighbor" begin
    X = [
        3 2 1
        2 2 1
        3 4 1
        2 2 2
        1 2 2
        1 3 2
    ]'
    # ideally, we would have transformed X before testing.
    tree = BruteTree(X)
    x = [1, 3, 2]
    k = 2                   
    all_neighbors = get_random_neighbor(X, tree, x; k, return_all_self = true)
    @test all_neighbors == X[:, [6, 5, 4]]
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
    all_pairwise_vdm = precompute_pairwise_value_difference(X, y)
    metric = ValueDifference(all_pairwise_vdm)
    tree = BruteTree(X', metric)
    k = 4

    new_point = generate_new_smoten_point(X', tree; k, rng)

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
    all_pairwise_vdm = precompute_pairwise_value_difference(X, y)
    smote_points = smoten_per_class(X', n, all_pairwise_vdm; k, rng)
    @test size(smote_points, 2) == n
end
