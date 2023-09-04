using Imbalance: rose, rose_per_class, ERR_NEG_S


# Test that it indeed generates n new points
@testset "rose_per_class" begin
    X = [1.0 1.0; 2.0 2.0; 3.0 3.0; 4.0 4.0; 5.0 5.0]'
    k = 10
    n = 100
    smote_points = rose_per_class(X, n; s = 1.0, rng = rng)
    @test size(smote_points, 2) == n
end

@testset "throws s error" begin
    X = [1.0 1.0; 2.0 2.0; 3.0 3.0; 4.0 4.0; 5.0 5.0]
    y = [0, 0, 0, 0, 0]
    k = 10
    n = 100
    s = -1.0
    @test_throws ERR_NEG_S(s) rose(X, y; s = s)
end

# Test that ROSE adds the right number of points per class and that the input and output types are as expected
@testset "ROSE Algorithm" begin
    tables = [
        "RowTable",
        "ColTable",
        "MatrixTable",
        "DictRowTable",
        "DictColTable",
        "Matrix",
    ]
    for i in eachindex(tables)
        @testset "ROSE with $tables[i] type" begin
            X, y = generate_imbalanced_data(
                1000,
                2;
                probs = [0.2, 0.6, 0.2],
                type = tables[i],
                rng = rng,
            )
            counts_per_class = countmap(y)
            majority_count = maximum(values(counts_per_class))
            Xover, yover =
                rose(X, y; s = 1.0, ratios = Dict(0 => 1.0, 1 => 1.2, 2 => 0.9), rng = rng)
            # if index is not 7 then return type must be a matrix table
            if i != 6
                @test Tables.istable(Xover)
                # convert to matrix so the following tests can proceed
                X = Tables.matrix(X)
                Xover = Tables.matrix(Xover)
                @test size(Xover, 1) == (
                    Int(round(1.0 * majority_count)) +
                    Int(round(1.2 * majority_count)) +
                    Int(round(0.9 * majority_count))
                )
                @test size(Xover, 1) == length(yover)
            else
                @test !Tables.istable(Xover) && isa(Xover, AbstractMatrix)
                @test size(Xover, 1) == (
                    Int(round(1.0 * majority_count)) +
                    Int(round(1.2 * majority_count)) +
                    Int(round(0.9 * majority_count))
                )
                @test size(Xover, 1) == length(yover)
            end
        end
    end
end



# Test that RNG can be int or Xoshiro of int in ROSE
@testset "RNG in ROSE Algorithm" begin
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
        @testset "ROSE with $tables[i] type" begin
            rng = Random.Xoshiro(1234)
            rng_int = 1234
            X, y = generate_imbalanced_data(
                100,
                2;
                probs = [0.2, 0.6, 0.2],
                type = tables[i],
                rng = rng,
            )
            rng = Random.Xoshiro(1234)
            Xover1, yover1 =
                rose(X, y; s = 0.03, ratios = Dict(0 => 1.0, 1 => 1.2, 2 => 0.9), rng = rng)
            Xover2, yover2 = rose(
                X,
                y;
                s = 0.03,
                ratios = Dict(0 => 1.0, 1 => 1.2, 2 => 0.9),
                rng = rng_int,
            )
            Xover3, yover3 =
                rose(X, y; s = 0.03, ratios = Dict(0 => 1.0, 1 => 1.2, 2 => 0.9), rng = 99)
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
@testset "materializer with rose" begin
    X, y =
        generate_imbalanced_data(1000, 2; probs = [0.2, 0.6, 0.2], type = "MatrixTable", rng = 121)
    Xover, yover = rose(DataFrame(X), y; ratios = Dict(0 => 1.0, 1 => 1.2, 2 => 0.9), rng = 121)
    # Check that the number of samples increased correctly
    @test typeof(Xover) == DataFrame
end
