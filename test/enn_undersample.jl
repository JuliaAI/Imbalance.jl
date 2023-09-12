using Imbalance: filter_modifier, compute_enn_filter, enn_undersample_per_class

@testset "filter computation, modification and application" begin

    # test computing the filter
	X = [1 2; 1.1 2.1; 1.05 2.05; 1.01 2.01; 2 3; 2.1 3.1; 2.05 3.05; 2.01 3.01]
	y = [1, 1, 1, 2, 2, 2, 2, 3]
	@test compute_enn_filter(X', y, 3, "mode") == [1, 1, 1, 0, 1, 1, 1, 0]

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
	y = [1, 1, 1, 2, 2, 2, 2, 2, 3, 2]
	@test compute_enn_filter(X', y, 4, "only mode") == [0, 0, 0, 0, 0, 1, 1, 1, 0, 1]
	@test compute_enn_filter(X', y, 4, "exists") == [1, 1, 1, 1, 1, 1, 1, 1, 0, 1]

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
	@test compute_enn_filter(X', y, 4, "all") == [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]

        # test filter modifier
	filter_modifier(BitVector([1, 1, 1, 1, 1, 0, 0, 0, 0, 0]), 5) ==
	[1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
	for n in 1:10
		@test sum(filter_modifier(BitVector([1, 1, 1, 1, 1, 0, 0, 0, 0, 0]), n, true)) == n
	end

    # test applying the filter
	@test enn_undersample_per_class(
		X'[:, 1:3],
		2,
		[1, 2, 3],
		[true, true, false, true, true, false, false, false, false, false],
	) == X[1:2, :]'

end



@testset "end-to-end test with imbalanced-learn" begin
    ENV["PYTHON"]=""
    import Pkg; Pkg.build("PyCall")
    using PyCall       
    using Conda    
    
    Conda.add("imbalanced-learn")
    Conda.add("numpy")
    Conda.add("scikit-learn")
    Conda.add("scipy")
    pyimport_conda("imblearn", "imbalanced-learn")
    pyimport_conda("numpy", "numpy")
    pyimport_conda("sklearn", "scikit-learn")
    pyimport_conda("scipy", "scipy")
    
    # Import numpy and other libraries
    np = pyimport("numpy")
    imblearn = pyimport("imblearn")
    fromsklearn = pyimport("sklearn.preprocessing")
    scipy_stats = pyimport("scipy.stats")

    # Set the dimensions of the matrix
    rows = 10000
    cols = 8

    # Generate a random matrix X and vector y in Python
    X = np.random.uniform(1.0, 10.0, size=(rows, cols))
    y = np.random.randint(1, 4, size=rows)

    # test that all keep condition works
    enn = imblearn.under_sampling.EditedNearestNeighbours(sampling_strategy="all", kind_sel="all")
    X_under1, y_under1 = enn.fit_resample(X, y)
    X_under2, y_under2 = enn_undersample(X, y; k = 3, keep_condition = "all", min_ratios = 0.01)

    @test sum(X_under1) ≈ sum(X_under2)
    @test sum(y_under1) ≈ sum(y_under2)

    # test that mode keep condition works 
    X = np.random.uniform(1.0, 10.0, size=(rows, cols))
    y = np.random.randint(1, 3, size=rows)      # Has to be 2, Scipy's mode is arbitrary when multiple ones exist

    enn = imblearn.under_sampling.EditedNearestNeighbours(sampling_strategy="all", kind_sel="mode")
    X_under1, y_under1 = enn.fit_resample(X, y)
    X_under2, y_under2 = enn_undersample(X, y; k = 3, keep_condition = "mode", min_ratios = 0.01, force_min_ratios = false)

    @test sum(X_under1) ≈ sum(X_under2)
    @test sum(y_under1) ≈ sum(y_under2)

    # test skipping a specific class 
    enn = imblearn.under_sampling.EditedNearestNeighbours(sampling_strategy="majority", kind_sel="mode")
    X_under1, y_under1 = enn.fit_resample(X, y)
    X_under2, y_under2 = enn_undersample(X, y; k = 3, keep_condition = "mode", min_ratios = Dict([mode(y)=>0.01]), force_min_ratios = false)
    @test sum(X_under1) ≈ sum(X_under2)
    @test sum(y_under1) ≈ sum(y_under2)

    # test min_ratios (already tested as a unit)
    X_under3, y_under3 = enn_undersample(X, y; k = 3, keep_condition = "mode", min_ratios = Dict([1=>0.98, 2=>0.98]), force_min_ratios = true)
    @test countmap(y_under3)[1] == countmap(y_under3)[2]
end