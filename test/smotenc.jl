using Imbalance:
    smotenc,
    smotenc_per_class,
    generate_new_smotenc_point,
    get_random_neighbor,
    get_neighbors_mode,
    get_penalty,
    EuclideanWithPenalty,
    get_cont_part,
    get_cat_part

@testset "Testing get_cont_part" begin
    # Test for vectors
    @test get_cont_part([1.0, 2.0, 3.0, "A", "B", "C"], 4) == [1.0, 2.0, 3.0]

    # Test for matrices
    @test get_cont_part(
        [
            1.0 2.0 3.0 4.0
            5.0 6.0 7.0 8.0
            "A" "B" "C" "D"
            "E" "F" "G" "H"
        ],
        3,
    ) == [
        1.0 2.0 3.0 4.0
        5.0 6.0 7.0 8.0
    ]
end

@testset "Testing get_cat_part" begin
    # Test for vectors
    @test get_cat_part([1.0, 2.0, 3.0, "A", "B", "C"], 3) == [3.0, "A", "B", "C"]

    # Test for matrices
    @test get_cat_part(
        [
            1.0 2.0 3.0 4.0
            5.0 6.0 7.0 8.0
            "A" "B" "C" "D"
            "E" "F" "G" "H"
        ],
        3,
    ) == [
        "A" "B" "C" "D"
        "E" "F" "G" "H"
    ]
end


@testset "Testing get_penalty" begin
    X = [
        1.0 2.0 3.0 4.0
        5.0 6.0 7.0 8.0
        4.5 7.3 1.2 0.1
        9.0 10.0 11.0 12.0
    ]
    split_ind = 4

    # get the std of the first three rows
    stds = std(X[1:3, :], dims = 2)
    # get the median of the stds
    median_std = median(stds)

    @test get_penalty(X, split_ind) ≈ median_std
end

@testset "Testing Distances.evaluate" begin

    x₁ = [1.0, 2.0, 3.0, 2, 3, 4]
    x₂ = [4.0, 5.0, 6.0, 2, 5, 4]
    split_ind = 4
    penalty = 0.5

    d = EuclideanWithPenalty(split_ind, penalty)

    @test Distances.evaluate(d, x₁, x₂) == sqrt(3^2 + 3^2 + 3^2) + penalty * 1
end


# Test that a random neighbor is indeed one of the nearest neighbors
@testset "get_random_neighbor" begin
    X = [
        100.0 100.0
        200.0 200.0
        3.5 3.5
        3.6 3.5
        4.0 4.0
        500.0 500.0
    ]'
    tree = BallTree(X)
    x = [3.5, 3.5]
    k = 2                   # wil become three in the function
    random_neighbor, all_neighbors = get_random_neighbor(X, tree, x; k, return_all = true)
    @test random_neighbor in [X[:, 4], X[:, 5]]
    @test all_neighbors == X[:, [4, 5]]
end

# Test that generated smote point is collinear with some pair of points 
# for the continuous part and is the mode for the categorical part
@testset "generate_new_smote_point" begin
    X = [
        1.0 1.0 9.7 3.3
        2.0 2.0 9.7 5.0
        3.0 3.0 9.7 5.0
        4.0 4.0 9.7 5.5
        5.0 5.0 1.2 5.0
    ]'
    tree = BallTree(X)
    k = 3
    split_ind = 3
    new_point = vec(generate_new_smotenc_point(X, tree, split_ind; k, rng))
    new_point_cont = get_cont_part(new_point, split_ind)
    Xcont = get_cont_part(X, split_ind)
    @test any(
        is_in_between(new_point_cont, Xcont[:, i], Xcont[:, j]) for
        i = 1:size(Xcont, 2), j = 1:size(Xcont, 2) if i != j
    )
    new_point_cat = get_cat_part(new_point, split_ind)
    @test new_point_cat == [9.7, 5.0]
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

    @test get_neighbors_mode(Xneighs, rng) == [1.0, 2.0, 3.0]
end



# Test that it indeed generates n new points
@testset "smote_per_class" begin
    X =
        [
            1.0 1.0 9.7 3.3
            2.0 2.0 9.7 5.0
            3.0 3.0 1.2 5.0
            4.0 4.0 3.3 5.5
            5.0 5.0 1.2 1.2
        ]'
    k = 3
    n = 100
    smote_points = smotenc_per_class(X, n, 3; k)
    @test size(smote_points, 2) == n
end
