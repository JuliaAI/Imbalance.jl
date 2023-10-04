using Imbalance:
    smote,
    smote_per_class,
    generate_new_smote_point,
    get_random_neighbor,
    get_collinear_point,
    ERR_NONPOS_K,
    WRN_K_TOO_BIG

# Test that a point is indeed collinear
@testset "get_collinear_point" begin
    x₁ = [1.0, 2.0, 3.0]
    x₂ = [4.0, 5.0, 6.0]
    collinear = vec(get_collinear_point(x₁, x₂; rng))
    @test is_in_between(collinear, x₁, x₂)
end

# Test that a random neighbor is indeed one of the nearest neighbors
@testset "get_random_neighbor" begin
    X = [100.0 100.0; 200.0 200.0; 3.5 3.5; 4.0 4.0; 500.0 500.0]'
    tree = KDTree(X)
    k = 2                   # wil become three in the function
    knn_map, _ = knn(tree, X, k + 1, true)
    ind = 3
    random_neighbor = get_random_neighbor(X, ind, knn_map; rng)
    @test random_neighbor in [X[:, 1], X[:, 4]]
end


# Test that generated smote point is collinear with some pair of points
@testset "generate_new_smote_point" begin
    X = [1.0 1.0; 2.0 2.0; 3.0 3.0; 4.0 4.0; 5.0 5.0]'
    tree = KDTree(X)
    k = 2
    knn_map, _ = knn(tree, X, k + 1, true)
    new_point = vec(generate_new_smote_point(X, knn_map; rng))
    @test any(
        is_in_between(new_point, X[:, i], X[:, j]) for
        i = 1:size(X, 2), j = 1:size(X, 2) if i != j
    )
end

# Test that it indeed generates n new points
@testset "smote_per_class" begin
    X = [1.0 1.0; 2.0 2.0; 3.0 3.0; 4.0 4.0; 5.0 5.0]
    k = 1
    n = 100
    smote_points = smote_per_class(X, n; k, rng)
    @test size(smote_points, 2) == n
end

# Test that it throws negative K error when needed
@testset "throws negative k error" begin
    X = [1.0 1.0; 2.0 2.0; 3.0 3.0; 4.0 4.0; 5.0 5.0]
    y = [1, 2, 3, 4, 5]
    k = -1
    n = 100
    @test_throws ERR_NONPOS_K(k) begin
        smote_per_class(X', n; k = k, rng)
    end
end

# Test that it throws k too big warning when needed
@testset "throws k too big warning" begin
    X = [1.0 1.0; 2.0 2.0; 3.0 3.0; 4.0 4.0; 5.0 5.0]
    y = [1, 1, 1, 1, 1]
    k = 10
    n = 100
    @test_logs (:warn, WRN_K_TOO_BIG(k, 5)) begin
        smote_per_class(X', n; k = k, rng)
    end
end



# Test that SMOTE adds the right number of points per class and that the input and output types are as expected
@testset "SMOTE Algorithm" begin
    tables = [
        "RowTable",
        "ColTable",
        "MatrixTable",
        "DictRowTable",
        "DictColTable",
        "Matrix",
        "MatrixTable",
    ]
    for i in eachindex(tables)
        @testset "SMOTE with $tables[i] type" begin
            X, y = generate_imbalanced_data(
                1000,
                2;
                class_probs = [0.2, 0.6, 0.2],
                type = tables[i],
                rng = rng,
            )
            counts_per_class = countmap(y)
            majority_count = maximum(values(counts_per_class))
            Xover, yover =
                smote(X, y; k = 5, ratios = Dict(0 => 1.0, 1 => 1.2, 2 => 0.9), rng = rng)
            # if index is not 7 then return type must be a matrix table
            if i != 6
                @test Tables.istable(Xover)
                # convert to matrix so the following tests can proceed
                X = Tables.matrix(X)
                Xover = Tables.matrix(Xover)
            else
                @test !Tables.istable(Xover) && isa(Xover, AbstractMatrix)
            end
            @test size(Xover, 1) == (
                Int(round(1.0 * majority_count)) +
                Int(round(1.2 * majority_count)) +
                Int(round(0.9 * majority_count))
            )
            @test size(Xover, 1) == length(yover)
        end
    end
end


# Test RNG for generate_new_smote_point, get_random_neighbor, get_collinear_point
@testset "RNG for Basic Functions" begin
    # check for consistency of results
    X = [1.0 1.0; 2.0 2.0; 3.0 3.0; 4.0 4.0; 5.0 5.0]'
    tree = KDTree(X)
    k = 2
    knn_map, _ = knn(tree, X, k + 1, true)
    rng = StableRNG(1234)
    random_neighbor1 = get_random_neighbor(X, 1, knn_map; rng)
    rng = StableRNG(1234)
    random_neighbor2 = get_random_neighbor(X, 1, knn_map; rng)
    @test random_neighbor1 == random_neighbor2
    rng = StableRNG(1234)
    collinear1 = vec(get_collinear_point(X[:, 1], X[:, 2]; rng))
    rng = StableRNG(1234)
    collinear2 = vec(get_collinear_point(X[:, 1], X[:, 2]; rng))
    @test collinear1 == collinear2
    rng = StableRNG(1234)
    new_point1 = vec(generate_new_smote_point(X, knn_map; rng))
    rng = StableRNG(1234)
    new_point2 = vec(generate_new_smote_point(X, knn_map; rng))
    @test new_point1 == new_point2
end

# Test that RNG can be int or StableRNG of int in SMOTE
@testset "RNG in SMOTE Algorithm" begin
    tables = [
        "RowTable",
        "ColTable",
        "MatrixTable",
        "DictRowTable",
        "DictColTable",
        "Matrix",
    ]
    for i in eachindex(tables)
        @testset "SMOTE with $tables[i] type" begin
            X, y = generate_imbalanced_data(
                100,
                2;
                class_probs = [0.2, 0.6, 0.2],
                type = tables[i],
                rng = rng,
            )
            Xover1, yover1 = smote(
                X,
                y;
                k = 5,
                ratios = Dict(0 => 1.0, 1 => 1.2, 2 => 0.9),
                rng = Random.Xoshiro(1234),
            )
            Xover2, yover2 =
                smote(X, y; k = 5, ratios = Dict(0 => 1.0, 1 => 1.2, 2 => 0.9), rng = 1234)
            Xover3, yover3 =
                smote(X, y; k = 5, ratios = Dict(0 => 1.0, 1 => 1.2, 2 => 0.9), rng = 99)
            if Tables.istable(X)
                Xover1 = Tables.matrix(Xover1)
                Xover2 = Tables.matrix(Xover2)
                Xover3 = Tables.matrix(Xover3)
            end
            @test sum(Xover1, dims = 1) == sum(Xover2, dims = 1)
            @test sum(Xover1, dims = 1) != sum(Xover3, dims = 1)
        end
    end
end


# test that the materializer works for dataframes
@testset "materializer" begin
    X, y =
        generate_imbalanced_data(1000, 2; class_probs = [0.2, 0.6, 0.2], type = "MatrixTable", rng = 121)
    Xover, yover = smote(DataFrame(X), y; ratios = Dict(0 => 1.0, 1 => 1.2, 2 => 0.9), rng = 121)
    # Check that the number of samples increased correctly
    @test typeof(Xover) == DataFrame
end
