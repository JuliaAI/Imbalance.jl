
using Imbalance: SMOTE, smote_per_class, get_midway_point, generate_new_smote_point, get_random_neighbor

rng = Random.default_rng(123)

# Test that a point is indeed midway
@testset "get_midway_point" begin
    x₁ = [1.0, 2.0, 3.0]
    x₂ = [4.0, 5.0, 6.0]
    midway = get_midway_point(x₁, x₂, rng)
    @test _is_in_between(midway, x₁ , x₂)
end

# Test that a random neighbor is indeed one of the nearest neighbors
@testset "get_random_neighbor" begin
    X = [100.0 100.0; 200.0 200.0; 3.5 3.5; 4.0 4.0; 500.0 500.0]'
    tree = KDTree(X)
    x = [3.4, 3.4]
    k = 2             # wil becomes three in the function
    random_neighbor = get_random_neighbor(X, tree, x, k, rng)
    @test random_neighbor in [X[:, 3], X[:, 4], X[:, 1]]
end


# Test that generated smote point is midway for some pair of points
@testset "generate_new_smote_point" begin
    X = [1.0 1.0; 2.0 2.0; 3.0 3.0; 4.0 4.0; 5.0 5.0]'
    tree = KDTree(X)
    k = 2
    new_point = generate_new_smote_point(X, tree, k, rng)
    @test any(_is_in_between(getobs(X, i), new_point, getobs(X, j)) for i in 1:numobs(X) for j in 1:numobs(X))
end

# Test that it indeed generates n new points
@testset "smote_per_class" begin
    X = [1.0 1.0; 2.0 2.0; 3.0 3.0; 4.0 4.0; 5.0 5.0]'
    k = 10
    n = 100
    smote_points = smote_per_class(X, n, k=k, rng=rng)
    @test numobs(smote_points) == n
end

# Test that SMOTE adds the right number of points per class and that the input and output types are as expected
@testset "SMOTE Algorithm" begin
    tables = ["DF", "RowTable", "ColTable", "MatrixTable", "DictRowTable", "DictColTable", "Matrix", "MatrixTable"]
    i = 1
    for table in tables
        @testset "SMOTE with $table type" begin
            X, y = generate_imbalanced_data(1000, 2; probs=[0.2, 0.6, 0.2], type=table, rng=rng)
            
            X_balanced, y_balanced = SMOTE(X, y; k=5, ratios=Dict(0=>1.0, 1=>1.2, 2=>0.9), rng=rng)
            # if index is not 7 then return type must be a matrix table
            if i != 7
                @test Tables.istable(X_balanced)
                # convert to matrix so the following tests can proceed
                X = Tables.matrix(X)'
                X_balanced = Tables.matrix(X_balanced)'
                # Does is return the right number of points per class? (We take into account that probs is not exact so check for 1% error)
                @test abs(numobs(X_balanced) - (0.6 * 1.0 * numobs(X) + 0.6 * 1.2 * numobs(X) + 0.6 * 0.9 * numobs(X))) < numobs(X) * 0.05
                @test numobs(X_balanced) == numobs(y_balanced)
            else
                @test !Tables.istable(X_balanced) && isa(X_balanced, AbstractMatrix)
                X, X_balanced = X', X_balanced'
                @test abs(numobs(X_balanced) - (0.6 * 1.0 * numobs(X) + 0.6 * 1.2 * numobs(X) + 0.6 * 0.9 * numobs(X))) < numobs(X) * 0.05
                @test numobs(X_balanced) == numobs(y_balanced)
            end
        end
        i += 1
    end
end

