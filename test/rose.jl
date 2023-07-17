using Imbalance: ROSE, rose_per_class


# Test that it indeed generates n new points
@testset "rose_per_class" begin
    X = [1.0 1.0; 2.0 2.0; 3.0 3.0; 4.0 4.0; 5.0 5.0]'
    k = 10
    n = 100
    smote_points = rose_per_class(X, n; s=1.0, rng=rng)
    @test numobs(smote_points) == n
    # check that there are no duplicates in the generated points
    @test length(unique(smote_points, dims=2)) == length(smote_points)
end


# Test that ROSE adds the right number of points per class and that the input and output types are as expected
@testset "ROSE Algorithm" begin
    tables = ["DF", "RowTable", "ColTable", "MatrixTable", "DictRowTable", "DictColTable", "Matrix", "MatrixTable"]
    i = 1
    for table in tables
        @testset "ROSE with $table type" begin
            X, y = generate_imbalanced_data(1000, 2; probs=[0.2, 0.6, 0.2], type=table, rng=rng)
            counts_per_class = group_counts(y)
            majority_count = maximum(values(counts_per_class))
            Xover, yover = ROSE(X, y; s=1.0, ratios=Dict(0=>1.0, 1=>1.2, 2=>0.9), rng=rng)
            # if index is not 7 then return type must be a matrix table
            if i != 7
                @test Tables.istable(Xover)
                # convert to matrix so the following tests can proceed
                X = Tables.matrix(X)'
                Xover = Tables.matrix(Xover)'
                # Does is return the right number of points per class? 
                @test numobs(Xover) == (Int(round(1.0 * majority_count)) + Int(round(1.2 * majority_count)) + Int(round(0.9 * majority_count)))
                @test numobs(Xover) == numobs(yover)
            else
                @test !Tables.istable(Xover) && isa(Xover, AbstractMatrix)
                X, Xover = X', Xover'
                @test numobs(Xover) == (Int(round(1.0 * majority_count)) + Int(round(1.2 * majority_count)) + Int(round(0.9 * majority_count)))
                @test numobs(Xover) == numobs(yover)
            end
        end
        i += 1
    end
end

