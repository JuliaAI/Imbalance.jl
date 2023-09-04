using Imbalance: rose, smote, random_oversample

using Test

RandomOversampler = Imbalance.TableTransforms.RandomOversampler
SMOTE = Imbalance.TableTransforms.SMOTE
ROSE = Imbalance.TableTransforms.ROSE

# Test isrevertible and isinvertible functions
@testset "isrevertible" begin
    @test isrevertible(ROSE) == true
    @test isrevertible(SMOTE) == true
    @test isrevertible(RandomOversampler) == true
end


@testset "TableTransforms" begin
    y_ind = 3
    smote_t = SMOTE(y_ind; k = 5, rng = 42)
    rose_t = ROSE(y_ind; s = 1.0, rng = 42)
    random_oversample_t = RandomOversampler(y_ind; rng = 42)
    oversample_funs = [random_oversample, rose, smote]
    oversample_ts = [random_oversample_t, rose_t, smote_t]
    tables = ["RowTable", "ColTable", "MatrixTable", "DictRowTable", "DictColTable"]

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
                Xyover1 = oversample_fun(Xy, y_ind; rng = 42)

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
