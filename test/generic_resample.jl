
using Imbalance: generic_undersample, generic_oversample, generic_clean_per_class, filter_modifier

@testset "generic_undersample" begin
    X, y = generate_imbalanced_data(
        1000,
        10;
        cat_feats_num_vals = [],
        probs = [0.7, 0.2, 0.1],
        type = "Matrix",                
        insert_y = nothing,
        rng = 41,
    )
    undersample_per_class(X_label, n) = X_label[:, 1:n]
    # classic test
    Xover, yover = generic_undersample(X, y, undersample_per_class)
    @test size(Xover, 1) == size(X[y.==2, :], 1) * 3
    # test that ϕ works (first, thid class missing from dict)
    Xover, yover = generic_undersample(X, y, undersample_per_class; ratios=Dict([1=>1.0]))
    @test size(Xover, 1) == size(X[y.==0, :], 1) + 2 * size(X[y.==2, :], 1)
end

using Imbalance: generic_oversample


@testset "generic_oversample" begin
    X, y = generate_imbalanced_data(
        1000,
        10;
        cat_feats_num_vals = [],
        probs = [0.7, 0.2, 0.1],
        type = "Matrix",                
        insert_y = nothing,
        rng =41,
    )
    oversample_per_class(X_label, n) = ones(size(X_label, 1), n)
    # classic test
    Xover, yover = generic_oversample(X, y, oversample_per_class; ratios=1.0)
    @test size(Xover, 1) == size(X, 1) +  size(X[y.==0, :], 1) * 2 - (size(X[y.==1, :], 1)+size(X[y.==2, :], 1))
end


@testset "generic cleaning methods" begin
	X = [
		1 2
		1.1 2.1
		1.05 2.05
		1.01 2.01
		1.011 2.011
		2 3
		2.1 3.1
		2.05 3.05
		2.01 3.01
		2.011 3.011
	]
	y = [1, 1, 1, 1, 1, 2, 2, 2, 3, 2]

    # test filter modifier
	filter_modifier(BitVector([1, 1, 1, 1, 1, 0, 0, 0, 0, 0]), 5) ==
	[1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
	for n in 1:10
		@test sum(filter_modifier(BitVector([1, 1, 1, 1, 1, 0, 0, 0, 0, 0]), n, true)) == n
	end

    # test applying the filter
	@test generic_clean_per_class(
		X'[:, 1:3],
		2,
		[1, 2, 3],
		[true, true, false, true, true, false, false, false, false, false],
	) == X[1:2, :]'

end