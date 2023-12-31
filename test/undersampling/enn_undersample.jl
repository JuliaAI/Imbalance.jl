using Imbalance: compute_enn_filter

@testset "filter computation" begin

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
end

@testset "end-to-end test with imbalanced-learn" begin
    if !offline_python_test
        ENV["PYTHON"] = ""
        import Pkg
        Pkg.build("PyCall")
        using PyCall
        using Conda
    
        Conda.add("imbalanced-learn=0.11")
        Conda.add("numpy=1.26")
        Conda.add("scikit-learn=1.3")
        Conda.add("scipy=1.11")
    
        # Import numpy and other libraries
        np = pyimport("numpy")
        imblearn = pyimport("imblearn")
        fromsklearn = pyimport("sklearn.preprocessing")
        scipy_stats = pyimport("scipy.stats")
    
        # Set the dimensions of the matrix
        rows = 10000
        cols = 8
    
        # Generate a random matrix X and vector y in Python
        X = np.random.uniform(1.0, 10.0, size = (rows, cols))
        y = np.random.randint(1, 4, size = rows)
    
        # test that all keep condition works
        enn = imblearn.under_sampling.EditedNearestNeighbours(
            sampling_strategy = "all",
            kind_sel = "all",
        )
    
        X_under1, y_under1 = enn.fit_resample(X, y)
        # Save  to a file
        write_var((X, y), "saved_vars/imblearn_enn_all_XY.jls")
        write_var((X_under1, y_under1), "saved_vars/imblearn_enn_all_XY_under.jls")
    
    else
        # Read from file
        (X, y) = read_var("saved_vars/imblearn_enn_all_XY.jls")
        (X_under1, y_under1) = read_var("saved_vars/imblearn_enn_all_XY_under.jls")
    end
    
        X_under2, y_under2 =
            enn_undersample(X, y; k = 3, keep_condition = "all", min_ratios = 0.01)
    
        @test sum(X_under1) ≈ sum(X_under2)
        @test sum(y_under1) ≈ sum(y_under2)
    
    
    if !offline_python_test
        # test that mode keep condition works 
        X = np.random.uniform(1.0, 10.0, size = (rows, cols))
        # Has to be 2, Scipy's mode is arbitrary when multiple ones exist
        y = np.random.randint(1, 3, size = rows)      
    
        enn = imblearn.under_sampling.EditedNearestNeighbours(
            sampling_strategy = "all",
            kind_sel = "mode",
        )
        X_under1, y_under1 = enn.fit_resample(X, y)
        write_var((X, y), "saved_vars/imblearn_enn_mode_XY.jls")
        write_var((X_under1, y_under1), "saved_vars/imblearn_enn_mode_XY_under.jls")
    else
         # Read from file
         (X, y) = read_var("saved_vars/imblearn_enn_mode_XY.jls")
         (X_under1, y_under1) = read_var("saved_vars/imblearn_enn_mode_XY_under.jls")
    end
    
        X_under2, y_under2 = enn_undersample(
            X,
            y;
            k = 3,
            keep_condition = "mode",
            min_ratios = 0.01,
            force_min_ratios = false,
        )
    
        @test sum(X_under1) ≈ sum(X_under2)
        @test sum(y_under1) ≈ sum(y_under2)
    
    if !offline_python_test
        # test skipping a specific class 
        enn = imblearn.under_sampling.EditedNearestNeighbours(
            sampling_strategy = "majority",
            kind_sel = "mode",
        )
        X_under1, y_under1 = enn.fit_resample(X, y)
        # Save  to a file
        write_var((X_under1, y_under1), "saved_vars/imblearn_enn_majority_XY_under.jls")
    else
        # Read from file
        (X_under1, y_under1) = read_var("saved_vars/imblearn_enn_majority_XY_under.jls")
    end
    
        X_under2, y_under2 = enn_undersample(
            X,
            y;
            k = 3,
            keep_condition = "mode",
            min_ratios = Dict([mode(y) => 0.01]),
            force_min_ratios = false,
        )
    
        @test sum(X_under1) ≈ sum(X_under2)
        @test sum(y_under1) ≈ sum(y_under2)
    
        # test min_ratios (already tested as a unit)
        X_under3, y_under3 = enn_undersample(
            X,
            y;
            k = 3,
            keep_condition = "mode",
            min_ratios = Dict([1 => 0.98, 2 => 0.98]),
            force_min_ratios = true,
        )
        @test countmap(y_under3)[1] == countmap(y_under3)[2]
end
