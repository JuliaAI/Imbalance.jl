
@testset "testing filtering logic" begin
    Xt = [   
        1 2 3 4 5
        1.1 2.1 3.1 4.1 5.1
        1.2 2.2 3.2 4.2 5.2
        1.3 2.3 3.3 4.3 5.3
        1.4 2.4 3.4 4.4 5.4
        1.5 2.5 3.5 4.5 5.5
        6 7 8 9 10
        6.1 7.1 8.1 9.1 10.1
        6.2 7.2 8.2 9.2 10.2
        6.3 7.3 8.3 9.3 10.3
        6.4 7.4 8.4 9.4 10.4
        6.5 7.5 8.5 9.5 10.5
        11 12 13 14 15
        11.1 12.1 13.1 14.1 15.1
        11.2 12.2 13.2 14.2 15.2
        11.3 12.3 13.3 14.3 15.3
        11.4 12.4 13.4 14.4 15.4
        11.5 12.5 13.5 14.5 15.5
    ]
    X = transpose(Xt)

    y = [0, 0, 0, 10, 10, 10, 1, 11, 11, 11, 11, 11, 2, 21, 21, 2, 2, 2]
    @test Imbalance.borderline1_filter(X, y; m=5) == [1, 1, 1, 1, 1 ,1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0]
end



@testset "end-to-end borderline smote test" begin
    X, y = Imbalance.generate_imbalanced_data(1000, 5; probs=[0.2, 0.8], type="Matrix", rng=Random.Xoshiro(42))
    # manual filter
    bool_filter = Imbalance.borderline1_filter(X', y; m=5)
    Xover, yover = borderline_smote1(X, y; k = 5, m = 5, ratios = 1.0, rng=Random.Xoshiro(42))

    # get filtered and unfilitered data
    X1, y1 = X[bool_filter, :], y[bool_filter]
    X0, y0 = X[.!bool_filter, :], y[.!bool_filter]
    # get oversampled data
    oversampled_range =  length(y) + 1:length(yover)
    X_ex, y_ex = Xover[oversampled_range, :], yover[oversampled_range]
    # test that SMOTE condition is met for all oversampled points
    for l in eachindex(y_ex)
        @test any(
        is_in_between(X_ex[l, :], X1[i, :], X1[j, :]) && y1[i] == y1[j] == y_ex[l] for
        i = 1:size(X1, 1), j = 1:size(X1, 1) if i != j
    )
    end
end


# Test that it throws k or m too big warning when needed
@testset "throws no borderline points" begin
    X = [1.0 1.0; 2.0 2.0; 2.2 2.2; 4.0 4.0; 5.0 5.0; 6.0 6.0]
    y = [1, 1, 1, 1, 1, 1]
    @test_throws Imbalance.ERR_NO_BORDERLINE begin
        Xover, yover = borderline_smote1(X, y; k = 2, m = 2, ratios = 1.3, rng=Random.Xoshiro(42), verbosity=0)
    end
    m = -5
    @test_throws Imbalance.ERR_NONPOS_M(m) begin
        Xover, yover = borderline_smote1(X, y; k = 2, m = m, ratios = 1.3, rng=Random.Xoshiro(42), verbosity=0)
    end
    y = [1, 1, 2, 2, 1, 1]
    m = 7
    @test_logs (:warn, Imbalance.WRN_M_TOO_BIG(m, 6)) begin
        Xover, yover = borderline_smote1(X, y; k = 1, m = m, ratios = 1.3, rng=Random.Xoshiro(42), verbosity=0)
    end
end

# test that the materializer works for dataframes
@testset "materializer" begin
    X, y =
        generate_imbalanced_data(1000, 2; probs = [0.2, 0.6, 0.2], type = "MatrixTable", rng = 121)
    Xover, yover = borderline_smote1(DataFrame(X), y; ratios = Dict(0 => 1.0, 1 => 1.2, 2 => 0.9), rng = 121)
    # Check that the number of samples increased correctly
    @test typeof(Xover) == DataFrame
end
