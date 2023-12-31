@testset "on average generates the same continuous point" begin
    X = [1.0 2.2 3.1 1.0]'
    R = Imbalance.XoshiroOrMT(42)
    Z = mean([Imbalance.generate_new_random_walk_point(X, [2, 3], [1, 4], [5., 8.], [[1.0], [1.0]]; rng=R) for i in 1:1000000])
    @test isapprox(collect(X), Z; rtol=0.01)
end

@testset "random walk per class end-to-end test" begin
    cont_inds = [1, 2, 3, 4, 5]
    cat_inds = [6, 7, 8]
    Xt, y = Imbalance.generate_imbalanced_data(100000, 5; num_vals_per_category=[3, 4, 2], type="Matrix")
    Xt[:, cont_inds] = Xt[:, cont_inds] .+ 0.4*randn(size(Xt, 1), 5) .+ 0.3
    X = transpose(Xt)
    
    R = Imbalance.XoshiroOrMT(42)

    X_new = Imbalance.random_walk_per_class(X, 100000, cont_inds, cat_inds; rng=R)

    # test proportions statistical property
    X_cat = Int32.(X[cat_inds, :])
    X_new_cat = Int32.(X_new[cat_inds, :])
    Q1 = [proportions(x) for x in eachrow(X_cat)]
    Q2 = [proportions(x) for x in eachrow(X_new_cat)]
    @test isapprox(Q1, Q2, rtol=0.01)

    X_cont = X[cont_inds, :]
    X_new_cont = X_new[cont_inds, :]

    # test mean statistical property
    M1 = vec(mean(X_cont, dims=2))
    M2 = vec(mean(X_new_cont, dims=2))
    @test isapprox(M1, M2, rtol=0.01)

    # test variance statistical property
    S1 = vec(mean(X_cont, dims=2))
    S2 = vec(mean(X_new_cont, dims=2))
    @test isapprox(S1, S2, rtol=0.01)
end


@testset "random walk oversample input types" begin
    # matrix input
    X, y = generate_imbalanced_data(
        1000,
        2;
        num_vals_per_category=[3],
        class_probs = [0.2, 0.6, 0.2],
        type = "Matrix",
        rng = 121,
    )
    counts_per_class = countmap(y)
    majority_count = maximum(values(counts_per_class))
    cat_inds = [3]
    Xover, yover =
        random_walk_oversample(X, y, cat_inds; ratios = Dict(0 => 1.0, 1 => 1.2, 2 => 0.9), rng = 121)
    # Check that the number of samples increased correctly
    @test size(Xover, 1) == (
        Int(round(1.0 * majority_count)) +
        Int(round(1.2 * majority_count)) +
        Int(round(0.9 * majority_count))
    )
    
    Xc, yc = generate_imbalanced_data(
        1000,
        2;
        class_probs = [0.2, 0.6, 0.2],
        type = "ColTable",
        rng = 121,
    )
    counts_per_class = countmap(yc)
    majority_count = maximum(values(counts_per_class))
    Xover, yover =
        random_walk_oversample(Xc, yc; ratios = Dict(0 => 1.0, 1 => 1.2, 2 => 0.9), rng = 121)
    # Check that the number of samples increased correctly
    @test Imbalance.rowcount(Xover) == (
        Int(round(1.0 * majority_count)) +
        Int(round(1.2 * majority_count)) +
        Int(round(0.9 * majority_count))
    )

end


# test that the materializer works for dataframes
@testset "materializer" begin
    X, y =
        generate_imbalanced_data(1000, 2; class_probs = [0.2, 0.6, 0.2], type = "MatrixTable", rng = 121)
    Xover, yover = random_walk_oversample(DataFrame(X), y; ratios = Dict(0 => 1.0, 1 => 1.2, 2 => 0.9), rng = 121)
    # Check that the number of samples increased correctly
    @test typeof(Xover) == DataFrame
end