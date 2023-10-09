using Imbalance:
    precompute_value_encodings,
    precompute_mvdm_distances,
    ValueDifference,
    EuclideanWithPenalty



@testset "MVDM Encoding and Distance" begin
    if !offline_python_test
        Conda.add("imbalanced-learn")
        Conda.add("numpy")
        Conda.add("scikit-learn")
    
        # Import necessary libraries
        np = pyimport("numpy")
        imblearn = pyimport("imblearn")
        fromsklearn = pyimport("sklearn.preprocessing")
    
        # Set the dimensions of the matrix
        rows = 50
        cols = 8
    
        # Generate a random matrix X and vector y in Python
        X = np.random.randint(1, 11, size=(rows, cols))
        y = np.random.randint(1, 6, size=rows)
    
        mvdm_encoder_python = imblearn.metrics.pairwise.ValueDifferenceMetric().fit(X, y)
        mvdm_encoder_python_proba_per_class_3 = mvdm_encoder_python.proba_per_class_[3]
    
        # Save mvdm_encoder_python to a file
        write_var(mvdm_encoder_python_proba_per_class_3, "saved_vars/mvdm_encoder_python.jls")
        write_var((X, y), "saved_vars/mvdm_encoder_python_XY.jls")
    else
        # Read mvdm_encoder_python from file
        mvdm_encoder_python_proba_per_class_3 = read_var("saved_vars/mvdm_encoder_python.jls")
        (X, y) = read_var("saved_vars/mvdm_encoder_python_XY.jls")
    end
    
    # Rest of your code (assuming mvdm_encoder_python is loaded or generated)...
    mvdm_encoder, num_cats_per_column = precompute_value_encodings(X, y)
    @test mvdm_encoder_python_proba_per_class_3[2:end,:] ≈ mvdm_encoder[3]'
    
    all_pairwise_mvdm = precompute_mvdm_distances(mvdm_encoder, num_cats_per_column)
    
    V = ValueDifference(all_pairwise_mvdm)
    D = pairwise(V, X, X, dims=1)
    @test sum(pairwise(V, X, X, dims=1)) ≈ sum(D)
end



@testset "penalized Euclidean metric" begin

    x₁ = [1.0, 2.0, 3.0, 2, 3, 4]
    x₂ = [4.0, 5.0, 6.0, 2, 5, 4]
    cont_inds = [1, 2, 3]
    cat_inds = [4, 5, 6]
    penalty = 0.5

    d = EuclideanWithPenalty(penalty, cont_inds, cat_inds)

    @test Distances.evaluate(d, x₁, x₂) ≈ 3^2 + 3^2 + 3^2 + penalty * 1
end


