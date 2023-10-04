using Imbalance:
    smote,
    rose,
    random_oversample,
    random_undersample,
    cluster_undersample,
    smotenc,
    smoten,
    random_walk_oversample,
    generate_imbalanced_data

@testset "Random Oversampler MLJ" begin
    failures, summary = MLJTestInterface.test(
        [Imbalance.MLJ.RandomOversampler],
        MLJTestInterface.make_multiclass()...;
        verbosity = 1,
        throw = true,
        mod = @__MODULE__
    )
    @test isempty(failures)
    num_rows = 100
    num_cont_feats = 5
    class_probs = [0.5, 0.2, 0.3]
    # table
    X, y = generate_imbalanced_data(num_rows, num_cont_feats; class_probs)
    X = DataFrame(X)
    model =
        Imbalance.MLJ.RandomOversampler(ratios = Dict(0 => 1.2, 1 => 1.2, 2 => 1.2), rng = 42)
    mach = machine(model)
    @test transform(mach, X, y) ==
          random_oversample(X, y; ratios = Dict(0 => 1.2, 1 => 1.2, 2 => 1.2), rng = 42)
    
    # matrix
    X, y = generate_imbalanced_data(num_rows, num_cont_feats; class_probs, type="Matrix")
    @test transform(mach, X, y) ==
          random_oversample(X, y; ratios = Dict(0 => 1.2, 1 => 1.2, 2 => 1.2), rng = 42)
end

@testset "ROSE MLJ" begin
    failures, summary = MLJTestInterface.test(
        [Imbalance.MLJ.ROSE],
        MLJTestInterface.make_multiclass()...;
        verbosity = 1,
        throw = true,
        mod = @__MODULE__
    )
    @test isempty(failures)
    num_rows = 100
    num_cont_feats = 5
    class_probs = [0.5, 0.2, 0.3]
    # table
    X, y = generate_imbalanced_data(num_rows, num_cont_feats; class_probs)
    X = DataFrame(X)
    model =
        Imbalance.MLJ.ROSE(s = 0.01,ratios = Dict(0 => 1.2, 1 => 1.2, 2 => 1.2), rng = 42)
    mach = machine(model)
    @test transform(mach, X, y) ==
          rose(X, y; s = 0.01, ratios = Dict(0 => 1.2, 1 => 1.2, 2 => 1.2), rng = 42)
    
    # matrix
    X, y = generate_imbalanced_data(num_rows, num_cont_feats; class_probs, type="Matrix")
    @test transform(mach, X, y) ==
        rose(X, y; s = 0.01, ratios = Dict(0 => 1.2, 1 => 1.2, 2 => 1.2), rng = 42)
end

@testset "SMOTE MLJ" begin
    failures, summary = MLJTestInterface.test(
        [Imbalance.MLJ.SMOTE],
        MLJTestInterface.make_multiclass()...;
        verbosity = 1,
        throw = true,
        mod = @__MODULE__
    )
    @test isempty(failures)
    num_rows = 100
    num_cont_feats = 5
    class_probs = [0.5, 0.2, 0.3]
    # table
    X, y = generate_imbalanced_data(num_rows, num_cont_feats; class_probs)
    X = DataFrame(X)
    model =
        Imbalance.MLJ.SMOTE(k = 5,ratios = Dict(0 => 1.2, 1 => 1.2, 2 => 1.2), rng = 42)
    mach = machine(model)
    @test transform(mach, X, y) ==
    smote(X, y; k = 5, ratios = Dict(0 => 1.2, 1 => 1.2, 2 => 1.2), rng = 42)

    # matrix
    X, y = generate_imbalanced_data(num_rows, num_cont_feats; class_probs, type="Matrix")
    @test transform(mach, X, y) ==
        smote(X, y; k = 5, ratios = Dict(0 => 1.2, 1 => 1.2, 2 => 1.2), rng = 42)
end


@testset "BorderlineSMOTE1 MLJ" begin
    failures, summary = MLJTestInterface.test(
        [Imbalance.MLJ.BorderlineSMOTE1],
        MLJTestInterface.make_multiclass()...;
        verbosity = 1,
        throw = true,
        mod = @__MODULE__
    )
    @test isempty(failures)
    num_rows = 500
    num_cont_feats = 4
    class_probs = [0.25, 0.5, 0.25]
    # table
    X, y = generate_imbalanced_data(num_rows, num_cont_feats; min_sep=0.0, class_probs, rng=42)

    X = DataFrame(X)
    model =
        Imbalance.MLJ.BorderlineSMOTE1(k = 5, m=6, ratios = Dict(0 => 1.2, 1 => 1.2, 2 => 1.2), rng = 42)
    mach = machine(model)
    @test transform(mach, X, y) ==
    borderline_smote1(X, y; k = 5, m=6, ratios = Dict(0 => 1.2, 1 => 1.2, 2 => 1.2), rng = 42)

    # matrix
    X, y = generate_imbalanced_data(num_rows, num_cont_feats; min_sep=0.0, class_probs, type="Matrix", rng=42)
    @test transform(mach, X, y) ==
        borderline_smote1(X, y; k = 5, m=6, ratios = Dict(0 => 1.2, 1 => 1.2, 2 => 1.2), rng = 42)
end

@testset "SMOTENC MLJ" begin
    failures, summary = MLJTestInterface.test(
        [Imbalance.MLJ.SMOTENC],
        MLJTestInterface.make_multiclass()...;
        verbosity = 1,
        throw = true,
        mod = @__MODULE__
    )
    @test isempty(failures)

    num_rows = 100
    num_cont_feats = 4
    class_probs = [0.5, 0.2, 0.3]

    num_vals_per_category = [3, 4, 2, 5]

    X, y = generate_imbalanced_data(num_rows, num_cont_feats; class_probs, num_vals_per_category)
    X = DataFrame(X)
    X = coerce(X, autotype(X, :few_to_finite))

    smotenc_model =
    Imbalance.MLJ.SMOTENC(k = 5, ratios = Dict(0 => 1.2, 1 => 1.2, 2 => 1.2), rng = 42)
    mach = machine(smotenc_model)
    @test transform(mach, X, y) ==
      smotenc(X, y; k = 5, ratios = Dict(0 => 1.2, 1 => 1.2, 2 => 1.2), rng = 42)
end


@testset "RandomWalkOversampler MLJ" begin
    failures, summary = MLJTestInterface.test(
        [Imbalance.MLJ.RandomWalkOversampler],
        MLJTestInterface.make_multiclass()...;
        verbosity = 1,
        throw = true,
        mod = @__MODULE__
    )
    @test isempty(failures)

    num_rows = 100
    num_cont_feats = 4
    class_probs = [0.5, 0.2, 0.3]

    num_vals_per_category = [3, 4, 2, 5]

    X, y = generate_imbalanced_data(num_rows, num_cont_feats; class_probs, num_vals_per_category)
    X = DataFrame(X)
    X = coerce(X, autotype(X, :few_to_finite))

    rwo_model =
    Imbalance.MLJ.RandomWalkOversampler(ratios = Dict(0 => 1.2, 1 => 1.2, 2 => 1.2), rng = 42)
    mach = machine(rwo_model)
    @test transform(mach, X, y) ==
      random_walk_oversample(X, y; ratios = Dict(0 => 1.2, 1 => 1.2, 2 => 1.2), rng = 42)
end

# For SMOTEN, need dataset with categorical variables. let's (perhaps) consider a PR later.
@testset "SMOTEN MLJ" begin
    num_rows = 100
    num_cont_feats = 0
    class_probs = [0.5, 0.2, 0.3]

    num_vals_per_category = [3, 4, 2, 5]
    # table
    X, y = generate_imbalanced_data(num_rows, num_cont_feats; class_probs, num_vals_per_category)
    X = DataFrame(X)
    X = coerce(X, autotype(X, :few_to_finite))
    smotenc_model =
        Imbalance.MLJ.SMOTEN(k = 5, ratios = Dict(0 => 1.2, 1 => 1.2, 2 => 1.2), rng = 42)
    mach = machine(smotenc_model)
    @test transform(mach, X, y) ==
          smoten(X, y; k = 5, ratios = Dict(0 => 1.2, 1 => 1.2, 2 => 1.2), rng = 42)

    # matrix
    X, y = generate_imbalanced_data(num_rows, num_cont_feats; class_probs, num_vals_per_category, type="Matrix")
    X = Int32.(X)
    @test transform(mach, X, y) ==
        smoten(X, y; k = 5, ratios = Dict(0 => 1.2, 1 => 1.2, 2 => 1.2), rng = 42)
end

@testset "Random Undersampler MLJ" begin
    failures, summary = MLJTestInterface.test(
        [Imbalance.MLJ.RandomUndersampler],
        MLJTestInterface.make_multiclass()...;
        verbosity = 1,
        throw = true,
        mod = @__MODULE__
    )
    @test isempty(failures)
    num_rows = 100
    num_cont_feats = 5
    class_probs = [0.5, 0.2, 0.3]
    X, y = generate_imbalanced_data(num_rows, num_cont_feats; class_probs)
    X = DataFrame(X)
    model =
        Imbalance.MLJ.RandomUndersampler(ratios = Dict(0 => 0.8, 1 => 0.8, 2 => 0.8), rng = 42)
    mach = machine(model)
    @test transform(mach, X, y) ==
    random_undersample(X, y; ratios = Dict(0 => 0.8, 1 => 0.8, 2 => 0.8), rng = 42)

    # matrix
    X, y = generate_imbalanced_data(num_rows, num_cont_feats; class_probs, type="Matrix")
    @test transform(mach, X, y) ==
        random_undersample(X, y; ratios = Dict(0 => 0.8, 1 => 0.8, 2 => 0.8), rng = 42)
end

@testset "Cluster Undersampler MLJ" begin
    failures, summary = MLJTestInterface.test(
        [Imbalance.MLJ.ClusterUndersampler],
        MLJTestInterface.make_multiclass()...;
        verbosity = 1,
        throw = true,
        mod = @__MODULE__
    )
    @test isempty(failures)
    num_rows = 100
    num_cont_feats = 5
    class_probs = [0.5, 0.2, 0.3]

    # table
    X, y = generate_imbalanced_data(num_rows, num_cont_feats; class_probs)
    X = DataFrame(X)
    model =
        Imbalance.MLJ.ClusterUndersampler(ratios = Dict(0 => 0.8, 1 => 0.8, 2 => 0.8), rng = 42)
    mach = machine(model)
    @test transform(mach, X, y) ==
          cluster_undersample(X, y; ratios = Dict(0 => 0.8, 1 => 0.8, 2 => 0.8), rng = 42)

    # matrix
    X, y = generate_imbalanced_data(num_rows, num_cont_feats; class_probs, type="Matrix")
    @test transform(mach, X, y) ==
          cluster_undersample(X, y; ratios = Dict(0 => 0.8, 1 => 0.8, 2 => 0.8), rng = 42)
end

@testset "ENN Undersampler MLJ" begin
    failures, summary = MLJTestInterface.test(
        [Imbalance.MLJ.ENNUndersampler],
        MLJTestInterface.make_multiclass()...;
        verbosity = 1,
        throw = true,
        mod = @__MODULE__
    )
    @test isempty(failures)
    num_rows = 100
    num_cont_feats = 5
    class_probs = [0.5, 0.2, 0.3]
    X, y = generate_imbalanced_data(num_rows, num_cont_feats; class_probs)
    X = DataFrame(X)
    model =
        Imbalance.MLJ.ENNUndersampler(min_ratios = Dict(0 => 0.8, 1 => 0.8, 2 => 0.8), rng = 42)
    mach = machine(model)
    @test transform(mach, X, y) ==
          enn_undersample(X, y; min_ratios = Dict(0 => 0.8, 1 => 0.8, 2 => 0.8), rng = 42)

    # matrix
    X, y = generate_imbalanced_data(num_rows, num_cont_feats; class_probs, type="Matrix")
    @test transform(mach, X, y) ==
          enn_undersample(X, y; min_ratios = Dict(0 => 0.8, 1 => 0.8, 2 => 0.8), rng = 42)
end

@testset "Tomek Undersampler MLJ" begin
    failures, summary = MLJTestInterface.test(
        [Imbalance.MLJ.TomekUndersampler],
        MLJTestInterface.make_multiclass()...;
        verbosity = 1,
        throw = true,
        mod = @__MODULE__
    )
    @test isempty(failures)
    num_rows = 100
    num_cont_feats = 5
    class_probs = [0.5, 0.2, 0.3]
    # table
    X, y = generate_imbalanced_data(num_rows, num_cont_feats; class_probs)
    X = DataFrame(X)
    model =
        Imbalance.MLJ.TomekUndersampler(min_ratios = Dict(0 => 0.8, 1 => 0.8, 2 => 0.8), rng = 42)
    mach = machine(model)
    @test transform(mach, X, y) ==
          tomek_undersample(X, y; min_ratios = Dict(0 => 0.8, 1 => 0.8, 2 => 0.8), rng = 42)

    # matrix
    X, y = generate_imbalanced_data(num_rows, num_cont_feats; class_probs, type="Matrix")
    @test transform(mach, X, y) ==
        tomek_undersample(X, y; min_ratios = Dict(0 => 0.8, 1 => 0.8, 2 => 0.8), rng = 42)
end