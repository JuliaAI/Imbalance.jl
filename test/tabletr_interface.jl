using Imbalance: ROSE_t, SMOTE_t, RandomOversampler_t, rose, smote, random_oversample

using Test


# Test isrevertible and isinvertible functions
@testset "isrevertible" begin
    @test isrevertible(SMOTE_t) == true
end


# Test that RNG can be int or StableRNG of int in SMOTE
@testset "TableTransforms" begin
    y_ind = 3
    smote_t = SMOTE_t(y_ind; k=5, rng=42)
    rose_t = ROSE_t(y_ind; s=1.0, rng=42)
    random_oversample_t = RandomOversampler_t(y_ind; rng=42)
    oversample_funs = [random_oversample, rose, random_oversample]
    oversample_ts = [random_oversample_t, rose_t, random_oversample_t]
    tables = [
        "DF",
        "RowTable",
        "ColTable",
        "MatrixTable",
        "DictRowTable",
        "DictColTable",
    ]

    for i in eachindex(tables)
        for (oversample_fun, oversample_t) in zip(oversample_funs, oversample_ts)
            Xy, _ = generate_imbalanced_data(
                50,
                4;
                probs = [0.25, 0.5, 0.25],
                type = tables[i],
                insert_y = y_ind,
                rng = 42,
            )
            @testset "TableTransform API with $tables[i] type" begin
                Xyover1 =
                    oversample_fun(Xy, y_ind; rng = 42)
                
                # apply works
                Xyover2, cache = apply(oversample_t, Xy)
                @test Tables.istable(Xyover1) == true
                @test Xyover1 == Xyover2

                # reapply works
                Xyover22, _ = reapply(oversample_t, Xy, cache)
                @test Xyover22 == Xyover2

                # revert works
                Xyover_i = revert(oversample_t, Xyover2, cache)
                @test Tables.matrix(Xyover_i) == Tables.matrix(Xy)
            end
        end
    end
end
