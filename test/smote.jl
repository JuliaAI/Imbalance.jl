using Imbalance: smote, smote_per_class, generate_new_smote_point, get_random_neighbor, 
    get_collinear_point

# Test that a point is indeed collinear
@testset "get_collinear_point" begin
    x₁ = [1.0, 2.0, 3.0]
    x₂ = [4.0, 5.0, 6.0]
    collinear = vec(get_collinear_point(x₁, x₂; rng))
    @test is_in_between(collinear, x₁ , x₂)
end

# Test that a random neighbor is indeed one of the nearest neighbors
@testset "get_random_neighbor" begin
    X = [100.0 100.0; 200.0 200.0; 3.5 3.5; 4.0 4.0; 500.0 500.0]
    tree = KDTree(X')
    x = [3.4, 3.4]
    k = 2                   # wil become three in the function
    random_neighbor = get_random_neighbor(X, tree, x; k, rng)
    @test random_neighbor in [X[3, :], X[4, :], X[1, :]]
end


# Test that generated smote point is collinear with some pair of points
@testset "generate_new_smote_point" begin
    X = [1.0 1.0; 2.0 2.0; 3.0 3.0; 4.0 4.0; 5.0 5.0]
    tree = KDTree(X')
    k = 2
    new_point = vec(generate_new_smote_point(X, tree; k, rng))
    @test any(is_in_between(new_point, X[i, :], X[j, :])
    for i in 1:size(X, 1), j in 1:size(X, 1) if i != j)
end

# Test that it indeed generates n new points
@testset "smote_per_class" begin
    X = [1.0 1.0; 2.0 2.0; 3.0 3.0; 4.0 4.0; 5.0 5.0]'
    k = 10
    n = 100
    smote_points = smote_per_class(X, n; k, rng)
    @test size(smote_points, 1) == n
end

# Test that SMOTE adds the right number of points per class and that the input and output types are as expected
@testset "SMOTE Algorithm" begin
    tables = ["DF", "RowTable", "ColTable", "MatrixTable", 
              "DictRowTable", "DictColTable", "Matrix", "MatrixTable"]
    for i in eachindex(tables)
        @testset "SMOTE with $tables[i] type" begin
            X, y = generate_imbalanced_data(1000, 2; probs=[0.2, 0.6, 0.2], 
                                            type=tables[i], rng=rng)
            counts_per_class = group_lens(y)
            majority_count = maximum(values(counts_per_class))
            Xover, yover = smote(X, y; k=5, ratios=Dict(0=>1.0, 1=>1.2, 2=>0.9), rng=rng)
            # if index is not 7 then return type must be a matrix table
            if i != 7
                @test Tables.istable(Xover)
                # convert to matrix so the following tests can proceed
                X = Tables.matrix(X)
                Xover = Tables.matrix(Xover)
            else
                @test !Tables.istable(Xover) && isa(Xover, AbstractMatrix)
            end
            @test size(Xover, 1) == (Int(round(1.0 * majority_count)) + 
            Int(round(1.2 * majority_count)) + 
            Int(round(0.9 * majority_count)))
            @test size(Xover, 1) == length(yover)
        end
    end
end

