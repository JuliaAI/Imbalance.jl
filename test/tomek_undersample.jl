
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

    # Set the dimensions of the matrix
    rows = 10000
    cols = 8

    # Generate a random matrix X and vector y in Python
    X = np.random.uniform(1.0, 10.0, size=(rows, cols))
    y = np.random.randint(1, 4, size=rows)


    # test that all keep condition works
    tl = imblearn.under_sampling.TomekLinks(sampling_strategy="all")
    X_res, y_res = tl.fit_resample(X, y)
    X_und, y_und = tomek_undersample(X, y; min_ratios=0.01, force_min_ratios=false)
    @test sum(X_res) ≈ sum(X_und)
    @test sum(y_res) ≈ sum(y_und)

    # test min_ratios (already tested as a unit)
    X_under3, y_under3 = tomek_undersample(X, y; min_ratios = Dict([1=>0.99, 2=>0.99]), force_min_ratios = true)
    @test countmap(y_under3)[1] == countmap(y_under3)[2]
end