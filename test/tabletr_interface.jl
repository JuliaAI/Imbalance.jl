using Imbalance:
    rose, smote, random_oversample, smoten, smotenc, random_undersample, cluster_undersample, random_walk_oversample

using Test

RandomOversampler = Imbalance.TableTransforms.RandomOversampler
SMOTE = Imbalance.TableTransforms.SMOTE
BorderlineSMOTE1 = Imbalance.TableTransforms.BorderlineSMOTE1
ROSE = Imbalance.TableTransforms.ROSE
SMOTEN = Imbalance.TableTransforms.SMOTEN
SMOTENC = Imbalance.TableTransforms.SMOTENC
RandomWalkOversampler = Imbalance.TableTransforms.RandomWalkOversampler
RandomUndersampler = Imbalance.TableTransforms.RandomUndersampler
ClusterUndersampler = Imbalance.TableTransforms.ClusterUndersampler
TomekUndersampler = Imbalance.TableTransforms.TomekUndersampler
ENNUndersampler = Imbalance.TableTransforms.ENNUndersampler

const oversampling_methods = [SMOTE, ROSE, SMOTEN, SMOTENC, RandomWalkOversampler, BorderlineSMOTE1]
const undersampling_methods = [RandomUndersampler, ClusterUndersampler, TomekUndersampler, ENNUndersampler]
# Test isrevertible and isinvertible functions
@testset "isrevertible" begin
    for method in oversampling_methods
        @test isrevertible(method) == true
        @test TransformsBase.isinvertible(method) == false
    end

    for method in undersampling_methods
        @test isrevertible(method) == false
        @test TransformsBase.isinvertible(method) == false
    end
end

function test_tabletr(oversample_fun, oversample_t, Xy, y_ind)
    Xyover1 = oversample_fun(Xy, y_ind; rng = 42)

    # apply works
    Xyover2, cache = apply(oversample_t, Xy)
    @test Tables.istable(Xyover1) == true
    @test Xyover1 == Xyover2

    # reapply works
    Xyover22, _ = reapply(oversample_t, Xy, cache)
    @test Xyover22 == Xyover2

    # revert works
    if isrevertible(oversample_t)
        Xyover_i = revert(oversample_t, Xyover2, cache)
        @test Tables.matrix(Xyover_i) == Tables.matrix(Xy)
    end
end

@testset "TableTransforms" begin
    y_ind = 5
    smote_t = SMOTE(y_ind; k = 5, rng = 42)
    borderline_smote_t = BorderlineSMOTE1(y_ind; k = 5, m = 5, rng = 42)
    rose_t = ROSE(y_ind; s = 1.0, rng = 42)
    random_oversample_t = RandomOversampler(y_ind; rng = 42)
    random_undersample_t = RandomUndersampler(y_ind; rng = 42)
    cluster_undersample_t = ClusterUndersampler(y_ind; rng = 42)
    enn_undersample_t = ENNUndersampler(y_ind; rng = 42)
    tomek_undersample_t = TomekUndersampler(y_ind; rng = 42)
    oversample_funs = [
        random_oversample,
        rose,
        smote,
        borderline_smote1,
        random_undersample,
        cluster_undersample,
        enn_undersample,
        tomek_undersample,
    ]
    oversample_ts = [
        random_oversample_t,
        rose_t,
        smote_t,
        borderline_smote_t,
        random_undersample_t,
        cluster_undersample_t,
        enn_undersample_t,
        tomek_undersample_t,
    ]
    tables = ["RowTable", "ColTable", "MatrixTable", "DictRowTable", "DictColTable"]

    for i in eachindex(tables)
        Xy, _ = generate_imbalanced_data(
            50,
            4;
            probs = [0.25, 0.5, 0.25],
            type = tables[i],
            insert_y = y_ind,
            rng = 42,
        )
        for (oversample_fun, oversample_t) in zip(oversample_funs, oversample_ts)
            @testset "TableTransform API with $tables[i] type" begin
                test_tabletr(oversample_fun, oversample_t, Xy, y_ind)
            end
        end

        @testset "TableTransform API with $tables[i] type for SMOTENC and SMOTEN and Random Walk" begin
            Xy, _ = generate_imbalanced_data(
                50,
                4;
                cat_feats_num_vals = [2, 6, 3, 3],
                probs = [0.25, 0.5, 0.25],
                type = tables[i],
                insert_y = y_ind,
                rng = 42,
            )
            Xy = coerce(Xy, autotype(Xy, :few_to_finite))
            test_tabletr(smotenc, SMOTENC(y_ind; k = 5, rng = 42), Xy, y_ind)
            test_tabletr(random_walk_oversample, RandomWalkOversampler(y_ind; rng = 42), Xy, y_ind)

            Xy, _ = generate_imbalanced_data(
                50,
                0;
                cat_feats_num_vals = [2, 4, 3, 3],
                probs = [0.25, 0.5, 0.25],
                type = tables[i],
                insert_y = y_ind,
                rng = 42,
            )
            Xy = coerce(Xy, autotype(Xy, :few_to_finite))
            test_tabletr(smoten, SMOTEN(y_ind; k = 5, rng = 42), Xy, y_ind)
        end
    end
end
